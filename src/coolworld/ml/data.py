from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import FeatureSchema

_ALLOWED_EVIDENCE_KINDS = {
    "fortyguard_live",
    "fortyguard_recorded_live",
    "city_open_data",
    "real_intervention_study",
    "public_observational_dataset",
}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    file_sha256: str
    source_records: tuple[dict[str, str], ...]
    schema: FeatureSchema

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = tuple(dict(x) for x in raw["source_records"])
        if not records:
            raise ValueError("dataset manifest must contain source_records")
        for record in records:
            if record.get("kind") not in _ALLOWED_EVIDENCE_KINDS:
                raise ValueError(f"unapproved evidence kind: {record.get('kind')}")
            digest = record.get("content_sha256", "")
            if len(digest) != 64:
                raise ValueError("every source record requires a SHA-256 content hash")
        schema = FeatureSchema(
            dynamic_features=tuple(raw["schema"]["dynamic_features"]),
            static_features=tuple(raw["schema"].get("static_features", [])),
            action_features=tuple(raw["schema"]["action_features"]),
            temperature_feature=raw["schema"].get("temperature_feature", "temperature_c"),
        )
        return cls(
            dataset_id=str(raw["dataset_id"]),
            file_sha256=str(raw["file_sha256"]),
            source_records=records,
            schema=schema,
        )


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class UrbanThermalSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Loads real, preprocessed spatiotemporal sequences from an NPZ bundle.

    Required arrays:
      dynamic: [S, T, N, Fd]
      static:  [S, N, Fs]
      actions: [S, T, N, Fa]
      mask:    [S, T, N] (True for valid tiles)

    No random/synthetic fallback exists. The NPZ hash must match the manifest.
    """

    def __init__(
        self,
        npz_path: str | Path,
        manifest_path: str | Path,
        *,
        context_len: int,
        pred_len: int,
    ) -> None:
        self.path = Path(npz_path)
        self.manifest = DatasetManifest.load(Path(manifest_path))
        if sha256_file(self.path) != self.manifest.file_sha256:
            raise ValueError("dataset file hash does not match manifest")
        if context_len <= 0 or pred_len <= 0:
            raise ValueError("context_len and pred_len must be positive")

        bundle = np.load(self.path, allow_pickle=False)
        required = {"dynamic", "static", "actions", "mask"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"dataset NPZ missing arrays: {sorted(missing)}")
        self.dynamic = np.asarray(bundle["dynamic"], dtype=np.float32)
        self.static = np.asarray(bundle["static"], dtype=np.float32)
        self.actions = np.asarray(bundle["actions"], dtype=np.float32)
        self.mask = np.asarray(bundle["mask"], dtype=bool)

        if self.dynamic.ndim != 4:
            raise ValueError("dynamic must have shape [S,T,N,F]")
        s, t, n, fd = self.dynamic.shape
        if self.static.shape[:2] != (s, n):
            raise ValueError("static shape must align with dynamic samples/tiles")
        if self.actions.shape[:3] != (s, t, n):
            raise ValueError("actions shape must align with dynamic")
        if self.mask.shape != (s, t, n):
            raise ValueError("mask shape must be [S,T,N]")
        if fd != len(self.manifest.schema.dynamic_features):
            raise ValueError("dynamic feature dimension differs from manifest schema")
        if self.static.shape[-1] != len(self.manifest.schema.static_features):
            raise ValueError("static feature dimension differs from manifest schema")
        if self.actions.shape[-1] != len(self.manifest.schema.action_features):
            raise ValueError("action feature dimension differs from manifest schema")
        if context_len + pred_len > t:
            raise ValueError("context_len + pred_len exceeds sequence length")
        if not np.all(np.isfinite(self.dynamic)):
            raise ValueError("dynamic array contains non-finite values")
        if not np.all(np.isfinite(self.static)):
            raise ValueError("static array contains non-finite values")
        if not np.all(np.isfinite(self.actions)):
            raise ValueError("actions array contains non-finite values")

        self.context_len = context_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return self.dynamic.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        c = self.context_len
        h = self.pred_len
        return {
            "context_dynamic": torch.from_numpy(self.dynamic[index, :c]),
            "context_actions": torch.from_numpy(self.actions[index, :c]),
            "context_mask": torch.from_numpy(self.mask[index, :c]),
            "future_dynamic": torch.from_numpy(self.dynamic[index, c : c + h]),
            "future_actions": torch.from_numpy(self.actions[index, c : c + h]),
            "future_mask": torch.from_numpy(self.mask[index, c : c + h]),
            "static": torch.from_numpy(self.static[index]),
        }
