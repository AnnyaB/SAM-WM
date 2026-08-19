from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from coolworld.interventions import Intervention

from .calibration import SupportConformalCalibrator
from .data import UrbanThermalSequenceDataset
from .future import extrapolate_time_features, known_future_indices
from .model import ActionConditionedJEPAWorldModel
from .support import local_action_support


@dataclass(frozen=True, slots=True)
class CounterfactualSummary:
    predicted_delta_c: float
    interval_low_c: float
    interval_high_c: float
    support_score: float
    status: str
    horizon_delta_c: tuple[float, ...]
    target_tile_ids: tuple[str, ...]
    target_horizon_delta_c: tuple[tuple[float, ...], ...]
    tile_ids: tuple[str, ...]
    future_timestamps: tuple[str, ...]
    baseline_temperature_c: tuple[tuple[float, ...], ...]
    candidate_temperature_c: tuple[tuple[float, ...], ...]


class CounterfactualInferenceEngine:
    def __init__(
        self, artifact_dir: str | Path, dataset_npz: str | Path, dataset_manifest: str | Path
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.manifest_path = Path(dataset_manifest)
        raw_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(
            self.artifact_dir / manifest["checkpoint"], map_location="cpu", weights_only=False
        )
        cfg = checkpoint["config"]
        self.context_len = int(cfg["context_len"])
        self.pred_len = int(cfg["pred_len"])
        self.dataset = UrbanThermalSequenceDataset(
            dataset_npz, dataset_manifest, context_len=self.context_len, pred_len=self.pred_len
        )
        schema = self.dataset.manifest.schema
        self.schema = schema
        self.known_idx = known_future_indices(schema)
        self.model = ActionConditionedJEPAWorldModel(
            len(schema.dynamic_features),
            len(schema.static_features),
            len(schema.action_features),
            len(self.known_idx),
            latent_dim=int(cfg["latent_dim"]),
            spatial_layers=int(cfg["spatial_layers"]),
            spatial_heads=int(cfg["spatial_heads"]),
            dropout=float(cfg["dropout"]),
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.tile_ids = tuple(str(x) for x in raw_manifest.get("tile_ids", []))
        self.grid_signature = str(raw_manifest.get("grid_signature", ""))
        if len(self.grid_signature) != 64:
            raise ValueError("dataset manifest grid_signature missing; rebuild with v0.4 ETL")
        if len(self.tile_ids) != self.dataset.dynamic.shape[2]:
            raise ValueError("dataset manifest tile_ids are required for intervention inference")
        cadence = raw_manifest.get("cadence_minutes_median")
        end = raw_manifest.get("timestamp_end")
        if cadence is None or end is None:
            raise ValueError(
                "dataset manifest needs cadence_minutes_median and timestamp_end; "
                "rebuild with v0.4 ETL"
            )
        self.cadence_minutes = float(cadence)
        self.last_timestamp = datetime.fromisoformat(str(end))
        cal = json.loads(
            (self.artifact_dir / "support_calibration.json").read_text(encoding="utf-8")
        )
        self.calibrator = SupportConformalCalibrator(
            bin_edges=np.asarray(cal["bin_edges"], dtype=float),
            quantiles=np.asarray(cal["quantiles"], dtype=float),
            alpha=float(cal["alpha"]),
        )

    def _support_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.context_len
        contexts = self.dataset.dynamic[:, :c].mean(axis=(1, 2))
        actions = self.dataset.actions[:, c : c + self.pred_len].mean(axis=(1, 2))
        return contexts, actions

    @torch.no_grad()
    def predict_latest(
        self, intervention: Intervention, *, grid_signature: str
    ) -> CounterfactualSummary:
        if grid_signature != self.grid_signature:
            raise ValueError("MODEL_CONTEXT_GRID_MISMATCH")
        target_idx = [self.tile_ids.index(x) for x in intervention.tile_ids if x in self.tile_ids]
        if len(target_idx) != len(intervention.tile_ids):
            return CounterfactualSummary(
                0,
                0,
                0,
                0,
                "UNKNOWN_INTERVENTION_TILE",
                tuple(),
                tuple(),
                tuple(),
                tuple(),
                tuple(),
                tuple(),
                tuple(),
            )

        # The last overlapping sequence contains the latest observed frames. Use
        # its final context_len frames and predict *beyond* timestamp_end.
        context_dynamic_np = self.dataset.dynamic[-1, -self.context_len :]
        context_actions_np = self.dataset.actions[-1, -self.context_len :]
        context_mask_np = self.dataset.mask[-1, -self.context_len :]
        static_np = self.dataset.static[-1]
        n = context_dynamic_np.shape[1]
        future_known_np, future_timestamps = extrapolate_time_features(
            self.last_timestamp,
            steps=self.pred_len,
            cadence_minutes=self.cadence_minutes,
            tiles=n,
            schema=self.schema,
        )
        baseline_actions_np = np.zeros(
            (self.pred_len, n, len(self.schema.action_features)), dtype=np.float32
        )
        candidate_actions_np = baseline_actions_np.copy()
        vec = intervention.action_vector()
        for i in target_idx:
            candidate_actions_np[:, i] = np.clip(candidate_actions_np[:, i] + vec, 0.0, 1.0)
        future_mask_np = np.repeat(context_mask_np[-1:, :], self.pred_len, axis=0)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)

        context_dynamic = t(context_dynamic_np)
        context_actions = t(context_actions_np)
        static = t(static_np)
        context_mask = torch.from_numpy(context_mask_np).unsqueeze(0)
        future_known = t(future_known_np)
        future_mask = torch.from_numpy(future_mask_np).unsqueeze(0)
        base = self.model(
            context_dynamic,
            context_actions,
            context_mask,
            static,
            t(baseline_actions_np),
            future_known,
            future_mask=future_mask,
        )
        cand = self.model(
            context_dynamic,
            context_actions,
            context_mask,
            static,
            t(candidate_actions_np),
            future_known,
            future_mask=future_mask,
        )
        base_temp = base.temperature_mean[0].cpu().numpy()
        cand_temp = cand.temperature_mean[0].cpu().numpy()
        delta_all = cand_temp - base_temp
        delta = delta_all[:, target_idx]
        horizon_delta = delta.mean(axis=1)
        mean_delta = float(delta.mean())

        contexts, actions = self._support_arrays()
        qctx = context_dynamic_np.mean(axis=(0, 1))
        support = local_action_support(
            qctx, contexts, actions, candidate_action=intervention.action_vector()
        )
        radius = self.calibrator.radius(support.support_score)
        status = "PREDICTED" if support.support_score >= 0.15 else "INSUFFICIENT_ACTION_SUPPORT"
        return CounterfactualSummary(
            predicted_delta_c=mean_delta,
            interval_low_c=mean_delta - radius,
            interval_high_c=mean_delta + radius,
            support_score=support.support_score,
            status=status,
            horizon_delta_c=tuple(float(x) for x in horizon_delta),
            target_tile_ids=tuple(intervention.tile_ids),
            target_horizon_delta_c=tuple(tuple(float(v) for v in row) for row in delta.T),
            tile_ids=self.tile_ids,
            future_timestamps=future_timestamps,
            baseline_temperature_c=tuple(tuple(float(v) for v in row) for row in base_temp),
            candidate_temperature_c=tuple(tuple(float(v) for v in row) for row in cand_temp),
        )
