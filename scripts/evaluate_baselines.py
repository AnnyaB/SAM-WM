from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coolworld.ml.baselines import linear_trend_forecast, persistence_forecast
from coolworld.ml.data import UrbanThermalSequenceDataset


def point_metrics(y_true: np.ndarray, mean: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)[mask]
    mu = np.asarray(mean, dtype=float)[mask]
    if y.size == 0:
        raise ValueError("no valid evaluation points")
    error = y - mu
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cheap forecasting baselines on the real SAM-WM sequence bundle."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--output", default="outputs/baselines.json")
    args = parser.parse_args()

    dataset = UrbanThermalSequenceDataset(
        args.dataset,
        args.manifest,
        context_len=args.context_len,
        pred_len=args.pred_len,
    )
    temp_idx = dataset.manifest.schema.temperature_index
    context = dataset.dynamic[:, : args.context_len]
    truth = dataset.dynamic[:, args.context_len : args.context_len + args.pred_len, :, temp_idx]
    mask = dataset.mask[:, args.context_len : args.context_len + args.pred_len]

    persistence = persistence_forecast(
        context,
        temperature_index=temp_idx,
        pred_len=args.pred_len,
    )
    trend = linear_trend_forecast(
        context,
        temperature_index=temp_idx,
        pred_len=args.pred_len,
    )

    payload = {
        "dataset_id": dataset.manifest.dataset_id,
        "evidence_only": True,
        "persistence": point_metrics(truth, persistence, mask),
        "linear_trend": point_metrics(truth, trend, mask),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
