from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from coolworld.ml.data import UrbanThermalSequenceDataset
from coolworld.ml.future import known_future_indices
from coolworld.ml.metrics import forecast_metrics
from coolworld.ml.model import ActionConditionedJEPAWorldModel
from coolworld.ml.ood import evaluate_temperature_tail


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained SAM-WM checkpoint on real extreme-temperature tails. "
            "This is a tail robustness test, not a claim of unseen-city or weather OOD."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/ood_eval.json")
    args = parser.parse_args()

    dataset = UrbanThermalSequenceDataset(
        args.dataset,
        args.manifest,
        context_len=args.context_len,
        pred_len=args.pred_len,
    )
    schema = dataset.manifest.schema
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    temp_idx = schema.temperature_index

    with torch.no_grad():
        for batch in loader:
            future_known = batch["future_dynamic"][..., list(known_idx)]
            output = model(
                batch["context_dynamic"],
                batch["context_actions"],
                batch["context_mask"],
                batch["static"],
                batch["future_actions"],
                future_known,
                future_mask=batch["future_mask"],
            )
            means.append(output.temperature_mean.numpy())
            stds.append(np.exp(output.temperature_log_scale.numpy()))
            truths.append(batch["future_dynamic"][..., temp_idx].numpy())
            masks.append(batch["future_mask"].numpy())

    mean = np.concatenate(means)
    std = np.concatenate(stds)
    truth = np.concatenate(truths)
    mask = np.concatenate(masks).astype(bool)
    overall = forecast_metrics(truth, mean, std, mask)

    payload = {
        "dataset_id": dataset.manifest.dataset_id,
        "scope": "real temperature tail robustness on this dataset",
        "warning": (
            "This does not establish unseen-city, unseen-climate, or causal "
            "intervention generalization. Those require separate held-out real "
            "datasets and real interventions."
        ),
        "overall": overall.__dict__,
        "extreme_heat_q95": evaluate_temperature_tail(truth, mean, std, mask, quantile=0.95),
        "extreme_heat_q99": evaluate_temperature_tail(truth, mean, std, mask, quantile=0.99),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
