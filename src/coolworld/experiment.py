from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .benchmarks import UrbanDataset
from .samwm import SAMWorldModel, SIGReg, samwm_loss


@dataclass(frozen=True)
class Normalizer:
    temp_mean: float
    temp_std: float
    rh_mean: float
    rh_std: float
    static_mean: tuple[float, float, float]
    static_std: tuple[float, float, float]


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    bias: float
    p95_abs_error: float
    conformal_90_coverage: float | None
    conformal_radius_c: float | None
    mean_surprise: float
    n_observed_targets: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def hours_since_origin(ts: np.ndarray, origin: np.datetime64) -> np.ndarray:
    # Absolute UTC phase keeps diurnal/annual calendar alignment under cross-city OOD transfer.
    epoch = np.datetime64("1970-01-01T00:00:00")
    return ((ts.astype("datetime64[s]") - epoch) / np.timedelta64(1, "h")).astype(np.float32)


def fit_normalizer(ds: UrbanDataset, train_mask: np.ndarray) -> Normalizer:
    t = ds.temperature[train_mask]
    rh = ds.rh[train_mask]
    temp_mean, temp_std = float(np.nanmean(t)), float(np.nanstd(t) + 1e-6)
    if np.isfinite(rh).any():
        rh_mean, rh_std = float(np.nanmean(rh)), float(np.nanstd(rh) + 1e-6)
    else:
        rh_mean, rh_std = 0.0, 1.0
    static = np.column_stack(
        [
            ds.lat,
            ds.lon,
            np.nan_to_num(
                ds.elevation,
                nan=float(np.nanmedian(ds.elevation)) if np.isfinite(ds.elevation).any() else 0.0,
            ),
        ]
    )
    sm, ss = np.mean(static, axis=0), np.std(static, axis=0) + 1e-6
    return Normalizer(
        temp_mean, temp_std, rh_mean, rh_std, tuple(map(float, sm)), tuple(map(float, ss))
    )


def normalized_dynamic(ds: UrbanDataset, norm: Normalizer) -> np.ndarray:
    temp = (ds.temperature - norm.temp_mean) / norm.temp_std
    rh = np.where(np.isfinite(ds.rh), (ds.rh - norm.rh_mean) / norm.rh_std, 0.0)
    return np.stack([temp, rh], axis=-1).astype(np.float32)


def normalized_static(ds: UrbanDataset, norm: Normalizer) -> np.ndarray:
    elev_fill = float(np.nanmedian(ds.elevation)) if np.isfinite(ds.elevation).any() else 0.0
    raw = np.column_stack([ds.lat, ds.lon, np.nan_to_num(ds.elevation, nan=elev_fill)]).astype(
        np.float32
    )
    return (
        (raw - np.asarray(norm.static_mean, np.float32)) / np.asarray(norm.static_std, np.float32)
    ).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        ds: UrbanDataset,
        norm: Normalizer,
        starts: np.ndarray,
        context: int,
        horizon: int,
        origin: np.datetime64,
    ) -> None:
        self.ds = ds
        self.norm = norm
        self.starts = starts.astype(int)
        self.context = context
        self.horizon = horizon
        self.origin = origin
        self.dynamic = normalized_dynamic(ds, norm)
        self.static = normalized_static(ds, norm)
        self.hours = hours_since_origin(ds.timestamps, origin)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        c = self.context
        h = self.horizon
        # Targets are physical Celsius; dynamic future is normalized only for latent JEPA target.
        return {
            "context_dynamic": torch.from_numpy(self.dynamic[s : s + c]),
            "static": torch.from_numpy(self.static),
            "context_time": torch.from_numpy(self.hours[s : s + c]),
            "future_time": torch.from_numpy(self.hours[s + c : s + c + h]),
            "future_dynamic": torch.from_numpy(self.dynamic[s + c : s + c + h]),
            "future_temp_c": torch.from_numpy(
                self.ds.temperature[s + c : s + c + h].astype(np.float32)
            ),
            "future_mask": torch.from_numpy(self.ds.observed_mask[s + c : s + c + h]),
        }


def make_starts(ds: UrbanDataset, start: str, end: str, context: int, horizon: int) -> np.ndarray:
    ts = ds.timestamps.astype("datetime64[ns]")
    lo, hi = np.datetime64(start), np.datetime64(end)
    starts = []
    for s in range(0, len(ts) - context - horizon + 1):
        target_start = ts[s + context]
        target_end = ts[s + context + horizon - 1]
        if target_start >= lo and target_end <= hi:
            starts.append(s)
    return np.asarray(starts, dtype=int)


def celsius_from_normalized(pred_norm: Tensor, norm: Normalizer) -> Tensor:
    return pred_norm * norm.temp_std + norm.temp_mean


def _forward_batch(
    model: SAMWorldModel,
    batch: dict[str, Tensor],
    ds: UrbanDataset,
    norm: Normalizer,
    device: torch.device,
    training: bool,
):
    b = {k: v.to(device) for k, v in batch.items()}
    out = model(
        b["context_dynamic"],
        b["static"],
        b["context_time"],
        b["future_time"],
        ds.edge_index.to(device),
        ds.edge_attr.to(device),
        future_dynamic_target=b["future_dynamic"] if training else None,
        future_temperature_target=(b["future_temp_c"] - norm.temp_mean) / norm.temp_std,
    )
    # Model evolves normalized temperature; convert output mean/log-scale to Celsius for metrics/NLL.
    mean_c = celsius_from_normalized(out.temperature_mean, norm)
    log_scale_c = out.temperature_log_scale + math.log(norm.temp_std)
    return out, mean_c, log_scale_c, b


def train_model(ds: UrbanDataset, cfg: dict, out_dir: str | Path, seed: int) -> Path:
    seed_everything(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    context, horizon = int(cfg["context_hours"]), int(cfg["horizon_hours"])
    train_starts = make_starts(
        ds, cfg["splits"]["train"][0], cfg["splits"]["train"][1], context, horizon
    )
    val_starts = make_starts(ds, cfg["splits"]["val"][0], cfg["splits"]["val"][1], context, horizon)
    if len(train_starts) == 0 or len(val_starts) == 0:
        raise ValueError("empty train/validation split")
    train_target_times = ds.timestamps[train_starts + context]
    train_mask = (ds.timestamps >= train_target_times.min()) & (
        ds.timestamps <= np.datetime64(cfg["splits"]["train"][1])
    )
    norm = fit_normalizer(ds, train_mask)
    origin = ds.timestamps[0]
    train_ds = WindowDataset(ds, norm, train_starts, context, horizon, origin)
    val_ds = WindowDataset(ds, norm, val_starts, context, horizon, origin)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    model = SAMWorldModel(
        hidden_dim=int(cfg["hidden_dim"]),
        max_source_step_c=float(cfg["max_source_step_normalized"]),
        residual_fraction=float(cfg["residual_fraction"]),
    ).to(device)
    sigreg = SIGReg(num_proj=int(cfg.get("sigreg_projections", 256))).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    best = float("inf")
    bad = 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out, mean_c, log_scale_c, b = _forward_batch(model, batch, ds, norm, device, True)
            # Loss in normalized space uses normalized target.
            target_norm = (b["future_temp_c"] - norm.temp_mean) / norm.temp_std
            loss, terms = samwm_loss(
                out, target_norm, b["future_mask"], sigreg, float(cfg["lambda_sig"])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_loader(model, val_loader, ds, norm, device, conformal_radius=None)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_mae_c": val_metrics.mae,
            "val_rmse_c": val_metrics.rmse,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics.mae < best - 1e-5:
            best = val_metrics.mae
            bad = 0
            ckpt = {
                "model": model.state_dict(),
                "config": cfg,
                "normalizer": asdict(norm),
                "seed": seed,
                "dataset": ds.name,
                "best_val_mae_c": best,
            }
            tmp = out_dir / "best.pt.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return out_dir / "best.pt"


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[SAMWorldModel, Normalizer, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = SAMWorldModel(
        hidden_dim=int(cfg["hidden_dim"]),
        max_source_step_c=float(cfg["max_source_step_normalized"]),
        residual_fraction=float(cfg["residual_fraction"]),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = Normalizer(**ck["normalizer"])
    return model, norm, ck


def conformal_radius(
    model: SAMWorldModel,
    loader: DataLoader,
    ds: UrbanDataset,
    norm: Normalizer,
    device: torch.device,
    alpha: float = 0.1,
) -> float:
    residuals = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out, mean_c, _, b = _forward_batch(model, batch, ds, norm, device, False)
            m = b["future_mask"]
            residuals.extend((b["future_temp_c"] - mean_c).abs()[m].cpu().numpy().tolist())
    if not residuals:
        raise ValueError("no observed calibration targets")
    r = np.asarray(residuals)
    q = min(1.0, math.ceil((len(r) + 1) * (1 - alpha)) / len(r))
    return float(np.quantile(r, q, method="higher"))


def evaluate_loader(
    model: SAMWorldModel,
    loader: DataLoader,
    ds: UrbanDataset,
    norm: Normalizer,
    device: torch.device,
    conformal_radius: float | None,
) -> Metrics:
    errors = []
    surprises = []
    covered = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out, mean_c, log_scale_c, b = _forward_batch(model, batch, ds, norm, device, False)
            m = b["future_mask"]
            e = (mean_c - b["future_temp_c"])[m]
            errors.extend(e.cpu().numpy().tolist())
            scale = log_scale_c.exp().clamp_min(1e-4)
            s = ((b["future_temp_c"] - mean_c).abs() / scale + log_scale_c)[m]
            surprises.extend(s.cpu().numpy().tolist())
            if conformal_radius is not None:
                covered.extend(
                    ((b["future_temp_c"] - mean_c).abs() <= conformal_radius)[m]
                    .cpu()
                    .numpy()
                    .tolist()
                )
    if not errors:
        raise ValueError("no observed targets for evaluation")
    e = np.asarray(errors, float)
    ae = np.abs(e)
    return Metrics(
        float(ae.mean()),
        float(np.sqrt(np.mean(e**2))),
        float(e.mean()),
        float(np.quantile(ae, 0.95)),
        None if conformal_radius is None else float(np.mean(covered)),
        conformal_radius,
        float(np.mean(surprises)),
        int(len(e)),
    )


def evaluate_split(
    ds: UrbanDataset,
    cfg: dict,
    checkpoint: str | Path,
    split: tuple[str, str],
    batch_size: int | None = None,
    radius: float | None = None,
) -> Metrics:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, norm, _ = load_checkpoint(checkpoint, device)
    context, horizon = int(cfg["context_hours"]), int(cfg["horizon_hours"])
    starts = make_starts(ds, split[0], split[1], context, horizon)
    windows = WindowDataset(ds, norm, starts, context, horizon, ds.timestamps[0])
    loader = DataLoader(
        windows, batch_size=batch_size or int(cfg["batch_size"]), shuffle=False, num_workers=0
    )
    return evaluate_loader(model, loader, ds, norm, device, radius)


def calibration_from_split(ds: UrbanDataset, cfg: dict, checkpoint: str | Path) -> float:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model, norm, _ = load_checkpoint(checkpoint, device)
    context, horizon = int(cfg["context_hours"]), int(cfg["horizon_hours"])
    starts = make_starts(ds, cfg["splits"]["val"][0], cfg["splits"]["val"][1], context, horizon)
    windows = WindowDataset(ds, norm, starts, context, horizon, ds.timestamps[0])
    loader = DataLoader(windows, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    return conformal_radius(model, loader, ds, norm, device, float(cfg.get("conformal_alpha", 0.1)))
