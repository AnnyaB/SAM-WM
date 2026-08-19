from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import UrbanThermalSequenceDataset
from .future import known_future_indices
from .losses import world_model_loss
from .model import ActionConditionedJEPAWorldModel


@dataclass(frozen=True, slots=True)
class TrainConfig:
    dataset_npz: str
    dataset_manifest: str
    output_dir: str = "artifacts/counterfactual_model"
    context_len: int = 12
    pred_len: int = 6
    latent_dim: int = 128
    spatial_layers: int = 2
    spatial_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.15
    seed: int = 42
    patience: int = 8


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _file_hash(path: Path) -> str:
    h = sha256(path.read_bytes()).hexdigest()
    return h


def train_world_model(cfg: TrainConfig) -> dict[str, object]:
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UrbanThermalSequenceDataset(
        cfg.dataset_npz, cfg.dataset_manifest, context_len=cfg.context_len, pred_len=cfg.pred_len
    )
    # The sequence builder uses stride 1, so adjacent samples share timestamps.
    # A random split would leak nearly identical windows across train/validation.
    # We therefore use a chronological validation block and purge L-1 windows
    # between train and validation so no source timestamp is shared.
    total = len(dataset)
    val_n = max(1, int(round(total * cfg.val_fraction)))
    val_start = total - val_n
    purge = cfg.context_len + cfg.pred_len - 1
    train_end = val_start - purge
    if train_end < 1:
        raise ValueError(
            "dataset too small for purged chronological train/validation split; "
            "collect more timestamps or reduce context/prediction length"
        )
    train_ds = Subset(dataset, range(0, train_end))
    val_ds = Subset(dataset, range(val_start, total))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    schema = dataset.manifest.schema
    known_idx = known_future_indices(schema)
    model = ActionConditionedJEPAWorldModel(
        len(schema.dynamic_features),
        len(schema.static_features),
        len(schema.action_features),
        len(known_idx),
        latent_dim=cfg.latent_dim,
        spatial_layers=cfg.spatial_layers,
        spatial_heads=cfg.spatial_heads,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    temp_idx = schema.temperature_index

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            future_known = batch["future_dynamic"][..., list(known_idx)]
            output = model(
                batch["context_dynamic"],
                batch["context_actions"],
                batch["context_mask"],
                batch["static"],
                batch["future_actions"],
                future_known,
                future_mask=batch["future_mask"],
                future_dynamic=batch["future_dynamic"],
            )
            loss = world_model_loss(
                output, batch["future_dynamic"][..., temp_idx], batch["future_mask"]
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            model.update_target_encoder()
            train_losses.append(float(loss.total.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                future_known = batch["future_dynamic"][..., list(known_idx)]
                output = model(
                    batch["context_dynamic"],
                    batch["context_actions"],
                    batch["context_mask"],
                    batch["static"],
                    batch["future_actions"],
                    future_known,
                    future_mask=batch["future_mask"],
                    future_dynamic=batch["future_dynamic"],
                )
                loss = world_model_loss(
                    output, batch["future_dynamic"][..., temp_idx], batch["future_mask"]
                )
                val_losses.append(float(loss.total.cpu()))
        row = {
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        history.append(row)
        if row["val_loss"] < best:
            best = row["val_loss"]
            stale = 0
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "schema": schema.to_dict()},
                out / "model.pt",
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    checkpoint = out / "model.pt"
    ckpt_hash = _file_hash(checkpoint)
    manifest = {
        "model_id": f"coolworld-sam-{ckpt_hash[:12]}",
        "checkpoint": "model.pt",
        "checkpoint_sha256": ckpt_hash,
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_sha256": dataset.manifest.file_sha256,
        "evidence": list(dataset.manifest.source_records),
        "schema": schema.to_dict(),
        "known_future_features": [schema.dynamic_features[i] for i in known_idx],
        "training": asdict(cfg),
        "best_validation_loss": best,
        "device_used": str(device),
        "split": {
            "policy": "chronological_purged",
            "train_sample_range": [0, train_end - 1],
            "purged_sample_range": [train_end, val_start - 1],
            "validation_sample_range": [val_start, total - 1],
            "purge_windows": purge,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return manifest
