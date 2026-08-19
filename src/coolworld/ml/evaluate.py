from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import UrbanThermalSequenceDataset
from .future import known_future_indices
from .metrics import forecast_metrics
from .model import ActionConditionedJEPAWorldModel


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_npz: str | Path,
    dataset_manifest: str | Path,
    *,
    context_len: int,
    pred_len: int,
    batch_size: int = 8,
) -> dict[str, object]:
    dataset = UrbanThermalSequenceDataset(
        dataset_npz, dataset_manifest, context_len=context_len, pred_len=pred_len
    )
    schema = dataset.manifest.schema
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    known_idx = known_future_indices(schema)
    model = ActionConditionedJEPAWorldModel(
        len(schema.dynamic_features),
        len(schema.static_features),
        len(schema.action_features),
        len(known_idx),
        latent_dim=int(cfg["latent_dim"]),
        spatial_layers=int(cfg["spatial_layers"]),
        spatial_heads=int(cfg["spatial_heads"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    means, stds, truths, masks = [], [], [], []
    temp_idx = schema.temperature_index
    with torch.no_grad():
        for batch in loader:
            future_known = batch["future_dynamic"][..., list(known_idx)]
            out = model(
                batch["context_dynamic"],
                batch["context_actions"],
                batch["context_mask"],
                batch["static"],
                batch["future_actions"],
                future_known,
                future_mask=batch["future_mask"],
            )
            means.append(out.temperature_mean.numpy())
            stds.append(np.exp(out.temperature_log_scale.numpy()))
            truths.append(batch["future_dynamic"][..., temp_idx].numpy())
            masks.append(batch["future_mask"].numpy())
    mean = np.concatenate(means)
    std = np.concatenate(stds)
    truth = np.concatenate(truths)
    mask = np.concatenate(masks).astype(bool)
    overall = forecast_metrics(truth, mean, std, mask)
    by_horizon = []
    for h in range(pred_len):
        m = forecast_metrics(truth[:, h], mean[:, h], std[:, h], mask[:, h])
        by_horizon.append({"horizon": h + 1, **m.__dict__})
    return {"overall": overall.__dict__, "by_horizon": by_horizon}
