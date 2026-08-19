from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from coolworld.ml.calibration import SupportConformalCalibrator
from coolworld.ml.data import UrbanThermalSequenceDataset
from coolworld.ml.future import known_future_indices
from coolworld.ml.model import ActionConditionedJEPAWorldModel
from coolworld.ml.support import local_action_support


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="artifacts/counterfactual_model/support_calibration.json")
    p.add_argument("--context-len", type=int, default=12)
    p.add_argument("--pred-len", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--bins", type=int, default=5)
    a = p.parse_args()
    ds = UrbanThermalSequenceDataset(
        a.dataset, a.manifest, context_len=a.context_len, pred_len=a.pred_len
    )
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    schema = ds.manifest.schema
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
    model.load_state_dict(ck["model"])
    model.eval()
    contexts = ds.dynamic[:, : a.context_len].mean(axis=(1, 2))
    actions = ds.actions[:, a.context_len : a.context_len + a.pred_len].mean(axis=(1, 2))
    residuals = []
    scores = []
    temp_idx = schema.temperature_index
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, b in enumerate(loader):
            future_known = b["future_dynamic"][..., list(known_idx)]
            out = model(
                b["context_dynamic"],
                b["context_actions"],
                b["context_mask"],
                b["static"],
                b["future_actions"],
                future_known,
                future_mask=b["future_mask"],
            )
            y = b["future_dynamic"][..., temp_idx]
            mask = b["future_mask"].bool()
            residuals.extend(torch.abs(y - out.temperature_mean)[mask].numpy().tolist())
            actual_action = actions[i]
            support = local_action_support(
                contexts[i], contexts, actions, candidate_action=actual_action
            )
            scores.extend([support.support_score] * int(mask.sum()))
    cal = SupportConformalCalibrator.fit(
        np.asarray(residuals), np.asarray(scores), alpha=a.alpha, bins=a.bins
    )
    payload = {
        "alpha": cal.alpha,
        "bin_edges": cal.bin_edges.tolist(),
        "quantiles": cal.quantiles.tolist(),
        "residual_count": len(residuals),
    }
    outp = Path(a.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
