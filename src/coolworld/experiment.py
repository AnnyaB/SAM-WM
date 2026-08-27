from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .benchmarks import UrbanDataset
from .graph import local_static_features
from .samwm import SAMWorldModel, SIGReg, samwm_loss


@dataclass(frozen=True)
class Normalizer:
    temp_mean: float
    temp_std: float
    rh_mean: float
    rh_std: float


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    bias: float
    p95_absolute_error: float
    conformal_coverage: float | None
    horizon_mae: tuple[float, ...]
    horizon_rmse: tuple[float, ...]
    mean_surprise: float
    parameter_count: int
    latency_ms_per_window: float
    n_observed_targets: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _time_mask(ds: UrbanDataset, split: tuple[str, str]) -> np.ndarray:
    timestamps = ds.timestamps.astype("datetime64[ns]")
    lo, hi = np.datetime64(split[0]), np.datetime64(split[1])
    return (timestamps >= lo) & (timestamps <= hi)


def fit_normalizer(ds: UrbanDataset, split: tuple[str, str]) -> Normalizer:
    time_mask = _time_mask(ds, split)
    temp = ds.temperature[time_mask]
    observed = ds.observed_mask[time_mask]
    valid_temp = temp[observed & np.isfinite(temp)]
    if valid_temp.size < 100:
        raise ValueError("insufficient observed training temperatures")
    rh = ds.rh[time_mask]
    valid_rh = rh[np.isfinite(rh)]
    temp_std = max(float(valid_temp.std()), 1e-4)
    if valid_rh.size:
        rh_mean, rh_std = float(valid_rh.mean()), max(float(valid_rh.std()), 1e-4)
    else:
        rh_mean, rh_std = 0.0, 1.0
    return Normalizer(float(valid_temp.mean()), temp_std, rh_mean, rh_std)


def normalized_dynamic(ds: UrbanDataset, norm: Normalizer) -> np.ndarray:
    temp = (ds.temperature - norm.temp_mean) / norm.temp_std
    rh_available = np.isfinite(ds.rh)
    rh = np.where(rh_available, (ds.rh - norm.rh_mean) / norm.rh_std, 0.0)
    return np.stack([temp, rh, rh_available.astype(np.float32)], axis=-1).astype(np.float32)


def normalized_static(ds: UrbanDataset) -> np.ndarray:
    return local_static_features(ds.lat, ds.lon, ds.elevation)


def derive_source_bound(
    ds: UrbanDataset,
    norm: Normalizer,
    split: tuple[str, str],
    *,
    quantile: float = 0.995,
) -> float:
    """Derive an admissible one-hour forcing scale from training observations only."""
    if not 0.9 <= quantile < 1:
        raise ValueError("source_bound_quantile must lie in [0.9,1)")
    mask = _time_mask(ds, split)
    temp = (ds.temperature - norm.temp_mean) / norm.temp_std
    valid_time = mask[:-1] & mask[1:]
    valid_obs = ds.observed_mask[:-1] & ds.observed_mask[1:]
    valid = valid_obs & valid_time[:, None] & np.isfinite(temp[:-1]) & np.isfinite(temp[1:])
    delta = np.abs(temp[1:] - temp[:-1])[valid]
    if delta.size < 100:
        raise ValueError("insufficient observed training deltas for source bound")
    bound = float(np.quantile(delta, quantile))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("invalid source bound derived from training data")
    return bound


def _hours(timestamps: np.ndarray) -> np.ndarray:
    base = np.datetime64("2000-01-01T00:00:00")
    return ((timestamps.astype("datetime64[s]") - base) / np.timedelta64(1, "h")).astype(np.float32)


def make_starts(
    ds: UrbanDataset,
    split: tuple[str, str],
    context: int,
    horizon: int,
) -> np.ndarray:
    if context < 1 or horizon < 1:
        raise ValueError("context/horizon must be >=1")
    ts = ds.timestamps.astype("datetime64[ns]")
    lo, hi = np.datetime64(split[0]), np.datetime64(split[1])
    starts = []
    for start in range(context, len(ts) - horizon + 1):
        target = ts[start : start + horizon]
        if target[0] >= lo and target[-1] <= hi:
            starts.append(start)
    if not starts:
        raise ValueError(f"no valid windows for split {split}")
    return np.asarray(starts, dtype=np.int64)


class WindowDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        ds: UrbanDataset,
        norm: Normalizer,
        split: tuple[str, str],
        context: int,
        horizon: int,
    ) -> None:
        self.ds = ds
        self.dynamic = normalized_dynamic(ds, norm)
        self.static = normalized_static(ds)
        self.hours = _hours(ds.timestamps)
        self.starts = make_starts(ds, split, context, horizon)
        self.context = context
        self.horizon = horizon
        self.temp_mean = norm.temp_mean
        self.temp_std = norm.temp_std

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = int(self.starts[index])
        c = slice(start - self.context, start)
        f = slice(start, start + self.horizon)
        target_c = ((self.ds.temperature[f] - self.temp_mean) / self.temp_std).astype(np.float32)
        return {
            "context_dynamic": torch.from_numpy(self.dynamic[c]),
            "static": torch.from_numpy(self.static),
            "context_time": torch.from_numpy(self.hours[c]),
            "future_time": torch.from_numpy(self.hours[f]),
            "future_dynamic": torch.from_numpy(self.dynamic[f]),
            "target_temperature": torch.from_numpy(target_c),
            "target_temperature_c": torch.from_numpy(self.ds.temperature[f].astype(np.float32)),
            "target_mask": torch.from_numpy(self.ds.observed_mask[f]),
        }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward_batch(
    model: SAMWorldModel,
    batch: dict[str, Tensor],
    edge_index: Tensor,
    edge_attr: Tensor,
    device: torch.device,
    *,
    training: bool,
):
    x = batch["context_dynamic"].to(device)
    static = batch["static"].to(device)
    ct = batch["context_time"].to(device)
    ft = batch["future_time"].to(device)
    target = batch["target_temperature"].to(device)
    future_dynamic = batch["future_dynamic"].to(device) if training else None
    return model(
        x,
        static,
        ct,
        ft,
        edge_index,
        edge_attr,
        future_dynamic_target=future_dynamic,
        future_temperature_target=target,
    )


def _resolved_config(ds: UrbanDataset, cfg: dict[str, Any], norm: Normalizer) -> dict[str, Any]:
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


def _build_model(cfg: dict[str, Any]) -> SAMWorldModel:
    return SAMWorldModel(
        dynamic_dim=int(cfg.get("dynamic_dim", 3)),
        static_dim=int(cfg.get("static_dim", 3)),
        hidden_dim=int(cfg["hidden_dim"]),
        max_source_step_normalized=float(cfg["max_source_step_normalized"]),
        residual_fraction=float(cfg["residual_fraction"]),
    )


def train_model(ds: UrbanDataset, cfg: dict[str, Any], out_dir: str | Path, seed: int) -> Path:
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_split = tuple(cfg["splits"]["train"])
    val_split = tuple(cfg["splits"]["val"])
    norm = fit_normalizer(ds, train_split)
    resolved = _resolved_config(ds, cfg, norm)
    train_ds = WindowDataset(
        ds, norm, train_split, int(cfg["context_hours"]), int(cfg["horizon_hours"])
    )
    val_ds = WindowDataset(
        ds, norm, val_split, int(cfg["context_hours"]), int(cfg["horizon_hours"])
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    device = _device()
    edge_index, edge_attr = ds.edge_index.to(device), ds.edge_attr.to(device)
    model = _build_model(resolved).to(device)
    sigreg = SIGReg(num_proj=int(cfg["sigreg_projections"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    best_mae = math.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    best_path = out_dir / "best.pt"
    for epoch in range(int(cfg["epochs"])):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = _forward_batch(model, batch, edge_index, edge_attr, device, training=True)
            target = batch["target_temperature"].to(device)
            mask = batch["target_mask"].to(device)
            loss, _ = samwm_loss(output, target, mask, sigreg, float(cfg["lambda_sig"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val = _evaluate_loader(model, val_loader, norm, ds, device, radius=None, latency=False)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae": val.mae}
        history.append(row)
        if val.mae < best_mae - 1e-6:
            best_mae = val.mae
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "normalizer": asdict(norm),
                    "config": resolved,
                    "seed": seed,
                    "dataset": ds.name,
                    "best_val_mae": best_mae,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return best_path


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    device = device or _device()
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = _build_model(cfg).to(device)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    norm = Normalizer(**blob["normalizer"])
    return model, norm, cfg, blob


def _evaluate_loader(
    model: SAMWorldModel,
    loader: DataLoader,
    norm: Normalizer,
    ds: UrbanDataset,
    device: torch.device,
    radius: float | None,
    *,
    latency: bool = True,
) -> Metrics:
    edge_index, edge_attr = ds.edge_index.to(device), ds.edge_attr.to(device)
    errors: list[np.ndarray] = []
    abs_by_horizon: list[list[np.ndarray]] | None = None
    sq_by_horizon: list[list[np.ndarray]] | None = None
    surprise_values: list[np.ndarray] = []
    cover = 0
    cover_n = 0
    observed_n = 0
    timings: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            start = time.perf_counter()
            output = _forward_batch(model, batch, edge_index, edge_attr, device, training=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            pred_c = output.temperature_mean.cpu().numpy() * norm.temp_std + norm.temp_mean
            target_c = batch["target_temperature_c"].numpy()
            mask = batch["target_mask"].numpy().astype(bool)
            err = pred_c - target_c
            errors.append(err[mask])
            observed_n += int(mask.sum())
            if abs_by_horizon is None:
                h = err.shape[1]
                abs_by_horizon = [[] for _ in range(h)]
                sq_by_horizon = [[] for _ in range(h)]
            for step in range(err.shape[1]):
                m = mask[:, step]
                if m.any():
                    abs_by_horizon[step].append(np.abs(err[:, step][m]))
                    sq_by_horizon[step].append(np.square(err[:, step][m]))
            if output.surprise is not None:
                s = output.surprise.cpu().numpy()
                surprise_values.append(s[mask])
            if radius is not None:
                cover += int((np.abs(err[mask]) <= radius).sum())
                cover_n += int(mask.sum())
    all_err = np.concatenate(errors) if errors else np.asarray([], float)
    if all_err.size == 0:
        raise ValueError("evaluation split has no observed targets")
    horizon_mae = tuple(
        float(np.concatenate(chunks).mean()) if chunks else float("nan")
        for chunks in (abs_by_horizon or [])
    )
    horizon_rmse = tuple(
        float(np.sqrt(np.concatenate(chunks).mean())) if chunks else float("nan")
        for chunks in (sq_by_horizon or [])
    )
    batch_size = max(1, int(getattr(loader, "batch_size", 1) or 1))
    latency_ms = (
        1000.0 * float(np.median(timings)) / batch_size if latency and timings else float("nan")
    )
    surprise = (
        float(np.concatenate(surprise_values).mean()) if surprise_values else float("nan")
    )
    return Metrics(
        mae=float(np.abs(all_err).mean()),
        rmse=float(np.sqrt(np.square(all_err).mean())),
        bias=float(all_err.mean()),
        p95_absolute_error=float(np.quantile(np.abs(all_err), 0.95)),
        conformal_coverage=(cover / cover_n if radius is not None and cover_n else None),
        horizon_mae=horizon_mae,
        horizon_rmse=horizon_rmse,
        mean_surprise=surprise,
        parameter_count=sum(p.numel() for p in model.parameters()),
        latency_ms_per_window=latency_ms,
        n_observed_targets=observed_n,
    )


def _prediction_errors(
    ds: UrbanDataset,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    split: tuple[str, str],
) -> np.ndarray:
    device = _device()
    model, norm, ckpt_cfg, _ = load_checkpoint(checkpoint, device)
    dataset = WindowDataset(
        ds,
        norm,
        split,
        int(ckpt_cfg["context_hours"]),
        int(ckpt_cfg["horizon_hours"]),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    edge_index, edge_attr = ds.edge_index.to(device), ds.edge_attr.to(device)
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            pred = _forward_batch(model, batch, edge_index, edge_attr, device, training=False)
            pred_c = pred.temperature_mean.cpu().numpy() * norm.temp_std + norm.temp_mean
            target_c = batch["target_temperature_c"].numpy()
            mask = batch["target_mask"].numpy().astype(bool)
            out.append(np.abs(pred_c - target_c)[mask])
    return np.concatenate(out)


def calibration_from_split(ds: UrbanDataset, cfg: dict[str, Any], checkpoint: str | Path) -> float:
    errors = _prediction_errors(ds, cfg, checkpoint, tuple(cfg["splits"]["val"]))
    if errors.size < 2:
        raise ValueError("not enough validation residuals for conformal calibration")
    alpha = float(cfg["conformal_alpha"])
    level = min(1.0, math.ceil((errors.size + 1) * (1 - alpha)) / errors.size)
    return float(np.quantile(errors, level, method="higher"))


def evaluate_split(
    ds: UrbanDataset,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    split: tuple[str, str],
    *,
    radius: float | None,
) -> Metrics:
    device = _device()
    model, norm, ckpt_cfg, _ = load_checkpoint(checkpoint, device)
    dataset = WindowDataset(
        ds,
        norm,
        split,
        int(ckpt_cfg["context_hours"]),
        int(ckpt_cfg["horizon_hours"]),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    return _evaluate_loader(model, loader, norm, ds, device, radius)
