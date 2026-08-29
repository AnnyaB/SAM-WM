from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .benchmarks import UrbanDataset, load_fairurbtemp, load_novisad
from .config import load_yaml
from .experiment import WindowDataset, derive_source_bound, fit_normalizer
from .freiburg import load_freiburg
from .paper_models import (
    BASELINES,
    PAPER_MODELS,
    batch_transform_for_variant,
    build_baseline,
    build_samwm,
    parameter_count,
    recommended_lambda_sig,
)
from .samwm import SAMWMOutput, SIGReg, samwm_loss

PAPER_SEEDS = (17, 29, 42, 73, 101)
DEADLINE_SEEDS = (17, 42, 73)


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    bias: float
    p95_absolute_error: float
    horizon_mae: tuple[float, ...]
    horizon_rmse: tuple[float, ...]
    conformal_coverage: float | None
    conformal_radius_c: float | None
    parameter_count: int
    latency_ms_per_window: float
    n_observed_targets: int


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def configure_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )


def _resolved_config(ds: UrbanDataset, cfg: dict[str, Any], norm) -> dict[str, Any]:
    resolved = dict(cfg)
    resolved["dynamic_dim"] = 3
    resolved["static_dim"] = 3
    resolved["edge_dim"] = 3
    resolved["max_source_step_normalized"] = derive_source_bound(
        ds,
        norm,
        tuple(cfg["splits"]["train"]),
        quantile=float(cfg.get("source_bound_quantile", 0.995)),
    )
    return resolved


def _build(name: str, resolved: dict[str, Any]) -> nn.Module:
    if name in BASELINES:
        return build_baseline(name, resolved)
    return build_samwm(resolved, name)


def _forward(
    name: str,
    model: nn.Module,
    batch: dict[str, Tensor],
    ds: UrbanDataset,
    device: torch.device,
    *,
    training: bool,
) -> Tensor | SAMWMOutput:
    transform = batch_transform_for_variant(name)
    batch = transform(batch)
    context = batch["context_dynamic"].to(device)
    static = batch["static"].to(device)
    if name == "itransformer":
        return model(context, static)
    if name == "timemixer":
        return model(context)
    future_dynamic = batch["future_dynamic"].to(device) if training else None
    return model(
        context,
        static,
        batch["context_time"].to(device),
        batch["future_time"].to(device),
        ds.edge_index.to(device),
        ds.edge_attr.to(device),
        future_dynamic_target=future_dynamic,
        future_temperature_target=batch["target_temperature"].to(device),
    )


def _prediction(output: Tensor | SAMWMOutput) -> Tensor:
    if isinstance(output, SAMWMOutput):
        return output.temperature_mean
    return output


def _masked_mae(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(pred.dtype)
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def _validation_mae(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    ds: UrbanDataset,
    norm,
    device: torch.device,
) -> float:
    total_abs = 0.0
    total_n = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = _forward(name, model, batch, ds, device, training=False)
            pred_c = _prediction(output).cpu().numpy() * norm.temp_std + norm.temp_mean
            target_c = batch["target_temperature_c"].numpy()
            mask = batch["target_mask"].numpy().astype(bool)
            diff = np.abs(pred_c - target_c)
            total_abs += float(diff[mask].sum())
            total_n += int(mask.sum())
    return total_abs / max(total_n, 1)


def _save_checkpoint(
    path: Path,
    *,
    name: str,
    seed: int,
    model: nn.Module,
    normalizer,
    resolved: dict[str, Any],
    best_val_mae: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "paper_suite_protocol": "SAM_WM_PAPER_SUITE_V1",
            "model_name": name,
            "seed": int(seed),
            "model": model.state_dict(),
            "normalizer": asdict(normalizer),
            "config": resolved,
            "best_val_mae": float(best_val_mae),
        },
        path,
    )


def _load_checkpoint(path: Path, name: str, device: torch.device):
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("paper_suite_protocol") != "SAM_WM_PAPER_SUITE_V1":
        raise RuntimeError(f"unexpected checkpoint protocol in {path}")
    if blob.get("model_name") != name:
        raise RuntimeError(f"checkpoint {path} is for {blob.get('model_name')}, not {name}")
    model = _build(name, blob["config"]).to(device)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model, blob


def train_one(
    *,
    name: str,
    seed: int,
    ds: UrbanDataset,
    cfg: dict[str, Any],
    out: Path,
    mode: str,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    configure_reproducibility(seed)
    train_split = tuple(cfg["splits"]["train"])
    val_split = tuple(cfg["splits"]["val"])
    norm = fit_normalizer(ds, train_split)
    resolved = _resolved_config(ds, cfg, norm)
    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    if mode == "deadline":
        epochs = min(epochs, 40)
        patience = min(patience, 6)
    context = int(cfg["context_hours"])
    horizon = int(cfg["horizon_hours"])
    train_ds = WindowDataset(ds, norm, train_split, context, horizon)
    val_ds = WindowDataset(ds, norm, val_split, context, horizon)
    train_loader = _loader(train_ds, int(cfg["batch_size"]), True, seed)
    val_loader = _loader(val_ds, int(cfg["batch_size"]), False, seed)
    run_dir = out / name / f"seed_{seed}"
    checkpoint = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    if resume and checkpoint.is_file() and history_path.is_file():
        return checkpoint, json.loads(history_path.read_text(encoding="utf-8"))

    device = _device()
    model = _build(name, resolved).to(device)
    sigreg = SIGReg(num_proj=int(cfg["sigreg_projections"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_mae = math.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward(name, model, batch, ds, device, training=True)
            target = batch_transform_for_variant(name)(batch)["target_temperature"].to(device)
            mask = batch["target_mask"].to(device)
            if isinstance(output, SAMWMOutput):
                loss, _ = samwm_loss(
                    output,
                    target,
                    mask,
                    sigreg,
                    recommended_lambda_sig(name, cfg),
                )
            else:
                loss = _masked_mae(output, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for {name} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_mae = _validation_mae(name, model, val_loader, ds, norm, device)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "val_mae_c": float(val_mae),
            }
        )
        if val_mae < best_mae - 1e-6:
            best_mae = val_mae
            stale = 0
            _save_checkpoint(
                checkpoint,
                name=name,
                seed=seed,
                model=model,
                normalizer=norm,
                resolved=resolved,
                best_val_mae=best_mae,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    if not checkpoint.is_file():
        raise RuntimeError(f"{name} seed {seed} produced no checkpoint")
    payload = {
        "protocol": "SAM_WM_PAPER_SUITE_TRAIN_V1",
        "name": name,
        "seed": seed,
        "mode": mode,
        "epochs_requested": epochs,
        "patience": patience,
        "best_val_mae_c": best_mae,
        "history": history,
    }
    _write_json(history_path, payload)
    return checkpoint, payload


def _finite_conformal_radius(errors: np.ndarray, alpha: float) -> float:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if errors.size < 2:
        raise ValueError("not enough calibration residuals")
    q = min(1.0, math.ceil((errors.size + 1) * (1.0 - alpha)) / errors.size)
    return float(np.quantile(errors, q, method="higher"))


def collect_absolute_errors(
    *,
    name: str,
    model: nn.Module,
    ds: UrbanDataset,
    norm,
    split: tuple[str, str],
    cfg: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    windows = WindowDataset(
        ds,
        norm,
        split,
        int(cfg["context_hours"]),
        int(cfg["horizon_hours"]),
    )
    loader = _loader(windows, int(cfg["batch_size"]), False, 0)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = _forward(name, model, batch, ds, device, training=False)
            pred_c = _prediction(output).cpu().numpy() * norm.temp_std + norm.temp_mean
            target_c = batch["target_temperature_c"].numpy()
            mask = batch["target_mask"].numpy().astype(bool)
            chunks.append(np.abs(pred_c - target_c)[mask])
    if not chunks:
        raise RuntimeError("calibration produced no residuals")
    return np.concatenate(chunks)


def evaluate_one(
    *,
    name: str,
    model: nn.Module,
    ds: UrbanDataset,
    norm,
    split: tuple[str, str],
    cfg: dict[str, Any],
    radius: float | None,
    device: torch.device,
) -> tuple[ForecastMetrics, dict[str, Any]]:
    windows = WindowDataset(
        ds,
        norm,
        split,
        int(cfg["context_hours"]),
        int(cfg["horizon_hours"]),
    )
    loader = _loader(windows, int(cfg["batch_size"]), False, 0)
    signed: list[np.ndarray] = []
    abs_h: list[list[np.ndarray]] = [[] for _ in range(int(cfg["horizon_hours"]))]
    sq_h: list[list[np.ndarray]] = [[] for _ in range(int(cfg["horizon_hours"]))]
    timings: list[tuple[float, int]] = []
    covered = 0
    covered_n = 0
    trace: dict[str, Any] | None = None
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = _forward(name, model, batch, ds, device, training=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            pred_c = _prediction(output).cpu().numpy() * norm.temp_std + norm.temp_mean
            target_c = batch["target_temperature_c"].numpy()
            mask = batch["target_mask"].numpy().astype(bool)
            timings.append((elapsed, int(pred_c.shape[0])))
            err = pred_c - target_c
            signed.append(err[mask])
            for h in range(pred_c.shape[1]):
                hm = mask[:, h]
                abs_h[h].append(np.abs(err[:, h])[hm])
                sq_h[h].append(np.square(err[:, h])[hm])
            if radius is not None:
                covered += int((np.abs(err[mask]) <= radius).sum())
                covered_n += int(mask.sum())
            if trace is None and batch_index == 0:
                trace = {
                    "target_mean_c": target_c[0].mean(axis=-1).astype(float).tolist(),
                    "prediction_mean_c": pred_c[0].mean(axis=-1).astype(float).tolist(),
                }
    all_err = np.concatenate(signed)
    horizon_mae = tuple(float(np.concatenate(x).mean()) for x in abs_h)
    horizon_rmse = tuple(float(np.sqrt(np.concatenate(x).mean())) for x in sq_h)
    total_time = sum(seconds for seconds, _ in timings)
    total_windows = sum(count for _, count in timings)
    metrics = ForecastMetrics(
        mae=float(np.abs(all_err).mean()),
        rmse=float(np.sqrt(np.square(all_err).mean())),
        bias=float(all_err.mean()),
        p95_absolute_error=float(np.quantile(np.abs(all_err), 0.95)),
        horizon_mae=horizon_mae,
        horizon_rmse=horizon_rmse,
        conformal_coverage=None if radius is None else covered / max(covered_n, 1),
        conformal_radius_c=radius,
        parameter_count=parameter_count(model),
        latency_ms_per_window=1000.0 * total_time / max(total_windows, 1),
        n_observed_targets=int(all_err.size),
    )
    return metrics, (trace or {})


def _source_normalizer(checkpoint: Path, name: str, device: torch.device):
    model, blob = _load_checkpoint(checkpoint, name, device)
    from .experiment import Normalizer

    return model, Normalizer(**blob["normalizer"]), blob["config"]


def _aggregate(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n_seeds": int(arr.size),
    }


def aggregate_results(raw: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, model_payload in raw.items():
        domains: dict[str, Any] = {}
        for domain, rows in model_payload["domains"].items():
            scalars = (
                "mae",
                "rmse",
                "bias",
                "p95_absolute_error",
                "conformal_coverage",
                "conformal_radius_c",
                "parameter_count",
                "latency_ms_per_window",
            )
            out_metrics: dict[str, Any] = {}
            for key in scalars:
                present = [row["metrics"].get(key) for row in rows]
                present = [float(value) for value in present if value is not None]
                if present:
                    out_metrics[key] = _aggregate(present)
            for key in ("horizon_mae", "horizon_rmse"):
                horizons = len(rows[0]["metrics"][key])
                out_metrics[key] = [
                    {
                        "horizon_hours": h + 1,
                        **_aggregate([float(row["metrics"][key][h]) for row in rows]),
                    }
                    for h in range(horizons)
                ]
            domains[domain] = {"metrics": out_metrics}
        summary[name] = {"domains": domains}
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    seeds = PAPER_SEEDS if args.mode == "paper" else DEADLINE_SEEDS
    models = tuple(args.models) if args.models else PAPER_MODELS
    unknown = sorted(set(models) - set(PAPER_MODELS))
    if unknown:
        raise SystemExit(f"unknown paper models: {unknown}")
    out = Path(args.out)
    source = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
    novisad = load_novisad(args.novisad_root, k=int(cfg["graph_k"]))
    fair = None
    if not args.skip_fairurb:
        if not args.fairurb_root:
            raise SystemExit("--fairurb-root is required unless --skip-fairurb is used")
        fair = load_fairurbtemp(args.fairurb_root, city=args.fairurb_city, k=int(cfg["graph_k"]))

    raw: dict[str, Any] = {}
    device = _device()
    for name in models:
        raw[name] = {"domains": {"freiburg": [], "novisad": []}}
        if fair is not None:
            raw[name]["domains"]["turku"] = []
        for seed in seeds:
            checkpoint, history = train_one(
                name=name,
                seed=seed,
                ds=source,
                cfg=cfg,
                out=out / "runs",
                mode=args.mode,
                resume=not args.no_resume,
            )
            model, norm, resolved = _source_normalizer(checkpoint, name, device)
            val_errors = collect_absolute_errors(
                name=name,
                model=model,
                ds=source,
                norm=norm,
                split=tuple(cfg["splits"]["val"]),
                cfg=resolved,
                device=device,
            )
            radius = _finite_conformal_radius(val_errors, float(cfg["conformal_alpha"]))
            domains: list[tuple[str, UrbanDataset, tuple[str, str]]] = [
                ("freiburg", source, tuple(cfg["splits"]["test"])),
                ("novisad", novisad, (str(novisad.timestamps[48]), str(novisad.timestamps[-1]))),
            ]
            if fair is not None:
                domains.append(
                    ("turku", fair, (str(fair.timestamps[48]), str(fair.timestamps[-1])))
                )
            for domain, dataset, split in domains:
                metrics, trace = evaluate_one(
                    name=name,
                    model=model,
                    ds=dataset,
                    norm=norm,
                    split=split,
                    cfg=resolved,
                    radius=radius,
                    device=device,
                )
                raw[name]["domains"][domain].append(
                    {
                        "seed": seed,
                        "metrics": asdict(metrics),
                        "trace": trace,
                        "best_val_mae_c": history["best_val_mae_c"],
                    }
                )
                print(
                    f"{name:24s} seed={seed:3d} {domain:8s} "
                    f"MAE={metrics.mae:.4f} RMSE={metrics.rmse:.4f} "
                    f"coverage={metrics.conformal_coverage:.4f}",
                    flush=True,
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload = {
        "protocol": "SAM_WM_PAPER_SUITE_V1",
        "mode": args.mode,
        "seeds": list(seeds),
        "models": list(models),
        "fairurb_city": None if fair is None else args.fairurb_city,
        "raw": raw,
        "summary": aggregate_results(raw),
    }
    _write_json(out / "paper_suite_results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train matched baselines/ablations and evaluate the SAM-WM paper suite."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--out", default="artifacts/paper_suite")
    parser.add_argument("--mode", choices=["paper", "deadline"], default="paper")
    parser.add_argument("--models", nargs="*", default=None, choices=PAPER_MODELS)
    parser.add_argument("--novisad-root", default="data/novisad")
    parser.add_argument("--fairurb-root", default=None)
    parser.add_argument("--fairurb-city", default="Turku")
    parser.add_argument("--skip-fairurb", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"status": "complete", "models": payload["models"]}, indent=2))


if __name__ == "__main__":
    main()
