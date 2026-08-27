from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from train import configure_reproducibility

import coolworld.experiment as experiment
from coolworld.benchmarks import UrbanDataset, load_freiburg, save_manifest
from coolworld.config import load_yaml
from coolworld.experiment import evaluate_split, make_starts, train_model
from coolworld.samwm import EDGE_DIM, SAMWMOutput, SAMWorldModel, SIGReg

RESEARCH_SEEDS = (17, 29, 42, 73, 101)
ABLATIONS = (
    "no_mental_map",
    "no_exchange",
    "unconstrained_exchange",
    "no_source_sink",
    "no_residual",
    "uniform_router",
    "no_temporal_memory",
)
CONTROLS = ("no_sigreg", "temperature_only")
PRE_FREEZE_STAGES = ("baselines", "full", "ablations", "controls")


class _IdentityMentalMap(nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        del edge_index, edge_attr
        return z


class _ZeroExchange(nn.Module):
    def forward(
        self,
        temp: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del z, edge_index, edge_attr, edge_weight
        return torch.zeros_like(temp), temp.new_zeros(())


class _UnconstrainedExchange(nn.Module):
    """Learned graph update without antisymmetry/conservation constraints."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim + EDGE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        temp: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        ea = edge_attr.unsqueeze(0).expand(z.shape[0], -1, -1)
        edge_delta = 0.45 * torch.tanh(
            self.update(torch.cat([z[:, src], z[:, dst], ea], dim=-1)).squeeze(-1)
        )
        edge_delta = edge_delta * edge_weight
        delta = temp.new_zeros(temp.shape)
        delta.index_add_(1, dst, edge_delta)
        conservation_error = delta.sum(dim=1).abs().mean()
        return delta, conservation_error


class _ZeroHead(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((*x.shape[:-1], 1))


class _UniformRouter(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((*x.shape[:-1], 4))


class _LastFrameTemporal(nn.Module):
    """Remove recurrent temporal memory while preserving tensor contracts."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x, x[:, -1].unsqueeze(0)


def apply_variant(model: SAMWorldModel, variant: str) -> SAMWorldModel:
    """Apply one named, retrainable structural ablation to a fresh model."""

    if variant == "full":
        return model
    if variant == "no_mental_map":
        model.mental_map = _IdentityMentalMap()
    elif variant == "no_exchange":
        model.exchange = _ZeroExchange()
    elif variant == "unconstrained_exchange":
        model.exchange = _UnconstrainedExchange(model.hidden_dim)
    elif variant == "no_source_sink":
        model.source = _ZeroHead()
    elif variant == "no_residual":
        model.residual = _ZeroHead()
    elif variant == "uniform_router":
        model.router = _UniformRouter()
    elif variant == "no_temporal_memory":
        model.temporal = _LastFrameTemporal()
    else:
        raise ValueError(f"unknown research variant: {variant}")
    return model


def _temperature_only_loss(
    output: SAMWMOutput,
    target_temperature: Tensor,
    target_mask: Tensor,
    sigreg: SIGReg,
    lambda_sig: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Control objective: temperature likelihood + same SIGReg, no latent prediction term."""

    mask = target_mask.to(target_temperature.dtype)
    denom = mask.sum().clamp_min(1.0)
    scale = output.temperature_log_scale.exp().clamp_min(1e-4)
    laplace_nll = (
        target_temperature - output.temperature_mean
    ).abs() / scale + output.temperature_log_scale
    temp_nll = (laplace_nll * mask).sum() / denom
    sig = sigreg(output.latent_pred[target_mask]) if target_mask.any() else temp_nll.new_zeros(())
    loss = temp_nll + float(lambda_sig) * sig
    zero = temp_nll.new_zeros(())
    return loss, {
        "loss": loss.detach(),
        "pred_loss": temp_nll.detach(),
        "latent": zero,
        "temperature_nll": temp_nll.detach(),
        "sigreg": sig.detach(),
    }


@contextmanager
def research_runtime(loss_variant: str = "standard") -> Iterator[None]:
    """Temporarily install research-only model variants without changing production code."""

    if loss_variant not in {"standard", "temperature_only"}:
        raise ValueError(f"unknown loss variant: {loss_variant}")

    original_build = experiment._build_model
    original_loss = experiment.samwm_loss

    def build_with_variant(cfg: dict[str, Any]) -> SAMWorldModel:
        model = original_build(cfg)
        return apply_variant(model, str(cfg.get("research_variant", "full")))

    experiment._build_model = build_with_variant
    if loss_variant == "temperature_only":
        experiment.samwm_loss = _temperature_only_loss

    try:
        yield
    finally:
        experiment._build_model = original_build
        experiment.samwm_loss = original_loss


def _metric_dict(error: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(error, dtype=float)
    error = error[np.isfinite(error)]
    if not error.size:
        raise ValueError("baseline produced no finite scored targets")
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
        "n_observed_targets": int(error.size),
    }


def evaluate_sanity_baselines(
    ds: UrbanDataset,
    cfg: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Validation-only non-trainable baselines with the exact SAM-WM window contract."""

    context = int(cfg["context_hours"])
    horizon = int(cfg["horizon_hours"])
    split = tuple(cfg["splits"]["val"])
    starts = make_starts(ds, split, context, horizon)

    errors: dict[str, list[np.ndarray]] = {
        "persistence": [],
        "linear_trend": [],
        "daily_persistence": [],
    }
    for start in starts:
        start = int(start)
        history = ds.temperature[start - context : start]
        target = ds.temperature[start : start + horizon]
        mask = ds.observed_mask[start : start + horizon].astype(bool)

        last = history[-1]
        previous = history[-2] if context >= 2 else last
        slope = last - previous

        predictions = {
            "persistence": np.repeat(last[None, :], horizon, axis=0),
            "linear_trend": np.stack(
                [last + (step + 1) * slope for step in range(horizon)], axis=0
            ),
            "daily_persistence": np.stack(
                [
                    history[-24 + step] if context >= 24 and step < 24 else last
                    for step in range(horizon)
                ],
                axis=0,
            ),
        }

        for name, prediction in predictions.items():
            valid = mask & np.isfinite(target) & np.isfinite(prediction)
            if valid.any():
                errors[name].append((prediction - target)[valid])

    return {name: _metric_dict(np.concatenate(chunks)) for name, chunks in errors.items() if chunks}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validation_metrics(
    ds: UrbanDataset,
    cfg: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    metrics = evaluate_split(
        ds,
        cfg,
        checkpoint,
        tuple(cfg["splits"]["val"]),
        radius=None,
    )
    return asdict(metrics)


def run_variant(
    ds: UrbanDataset,
    base_cfg: dict[str, Any],
    root: Path,
    *,
    name: str,
    seed: int,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    loss_variant = "standard"
    research_variant = name

    if name == "full":
        research_variant = "full"
    elif name == "no_sigreg":
        research_variant = "full"
        cfg["lambda_sig"] = 0.0
    elif name == "temperature_only":
        research_variant = "full"
        loss_variant = "temperature_only"
    elif name not in ABLATIONS:
        raise ValueError(f"unknown experiment: {name}")

    cfg["research_variant"] = research_variant
    cfg["loss_variant"] = loss_variant
    out = root / name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(ds, out / "dataset_manifest.json")

    with research_runtime(loss_variant):
        checkpoint = train_model(ds, cfg, out, seed)
        metrics = _validation_metrics(ds, cfg, checkpoint)

    payload = {
        "name": name,
        "research_variant": research_variant,
        "loss_variant": loss_variant,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "validation": metrics,
        "heldout_or_ood_accessed": False,
    }
    _write_json(out / "validation_metrics.json", payload)
    return payload


def run_stage(
    ds: UrbanDataset,
    cfg: dict[str, Any],
    root: Path,
    stage: str,
) -> None:
    if stage == "baselines":
        payload = {
            "split": "validation",
            "heldout_or_ood_accessed": False,
            "metrics": evaluate_sanity_baselines(ds, cfg),
        }
        _write_json(root / "baselines" / "validation_metrics.json", payload)
        return

    names: tuple[str, ...]
    if stage == "full":
        names = ("full",)
    elif stage == "ablations":
        names = ABLATIONS
    elif stage == "controls":
        names = CONTROLS
    else:
        raise ValueError(f"unknown stage: {stage}")

    for name in names:
        for seed in RESEARCH_SEEDS:
            result = run_variant(ds, cfg, root, name=name, seed=seed)
            print(
                json.dumps(
                    {
                        "name": name,
                        "seed": seed,
                        "val_mae": result["validation"]["mae"],
                    }
                ),
                flush=True,
            )


def write_pre_freeze_manifest(
    root: Path,
    *,
    config_path: Path,
    dataset_name: str,
) -> None:
    expected = [root / "baselines" / "validation_metrics.json"]
    expected.extend(
        root / name / f"seed_{seed}" / "validation_metrics.json"
        for name in ("full", *ABLATIONS, *CONTROLS)
        for seed in RESEARCH_SEEDS
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "pre-freeze research evidence is incomplete; refusing to seal manifest: "
            + ", ".join(missing)
        )
    evidence = sorted(expected)
    manifest = {
        "protocol": "SAM_WM_PRE_FREEZE_RESEARCH_V1",
        "dataset": dataset_name,
        "seeds": list(RESEARCH_SEEDS),
        "ablations": list(ABLATIONS),
        "controls": list(CONTROLS),
        "config_sha256": _sha256_file(config_path),
        "research_py_sha256": _sha256_file(Path(__file__)),
        "validation_artifacts": {
            str(path.relative_to(root)): _sha256_file(path) for path in evidence
        },
        "heldout_or_ood_accessed": False,
        "rule": (
            "This manifest contains development/validation evidence only. "
            "Final-test and OOD labels are forbidden until the reported branch is frozen."
        ),
    }
    _write_json(root / "PRE_FREEZE_MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM-WM validation-only baselines, ablations and controls before freeze."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument(
        "--stage",
        choices=(*PRE_FREEZE_STAGES, "all-pre-freeze"),
        default="all-pre-freeze",
    )
    parser.add_argument("--out", default="artifacts/research")
    args = parser.parse_args()

    configure_reproducibility()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    ds = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    stages = PRE_FREEZE_STAGES if args.stage == "all-pre-freeze" else (args.stage,)
    for stage in stages:
        run_stage(ds, cfg, root, stage)

    if args.stage == "all-pre-freeze":
        write_pre_freeze_manifest(
            root,
            config_path=config_path,
            dataset_name=ds.name,
        )


if __name__ == "__main__":
    main()
