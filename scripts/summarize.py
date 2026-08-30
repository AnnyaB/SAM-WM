from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    "mae",
    "rmse",
    "bias",
    "p95_absolute_error",
    "conformal_coverage",
    "mean_surprise",
    "latency_ms_per_window",
    "parameter_count",
)


def _aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/eval")
    parser.add_argument("--out", default="artifacts/summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    records: dict[str, list[dict]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("seed_*/*_metrics.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        evaluation = obj.get("evaluation") or f"{obj['dataset']}_{obj.get('split', 'unknown')}"
        records.setdefault(evaluation, []).append(obj["metrics"])
        metadata[evaluation] = {
            "dataset": obj["dataset"],
            "split": obj.get("split", "unknown"),
            "protocol": obj["protocol"],
        }

    if not records:
        raise SystemExit(f"no metrics under {root}/seed_*")

    summary: dict[str, dict] = {}
    for evaluation, rows in sorted(records.items()):
        entry: dict = {"metadata": metadata[evaluation], "metrics": {}}
        for key in METRICS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if values:
                entry["metrics"][key] = _aggregate(values)

        horizon_count = max((len(row.get("horizon_mae", [])) for row in rows), default=0)
        if horizon_count:
            entry["metrics"]["horizon_mae"] = []
            entry["metrics"]["horizon_rmse"] = []
            for step in range(horizon_count):
                for key in ("horizon_mae", "horizon_rmse"):
                    values = [
                        float(row[key][step])
                        for row in rows
                        if len(row.get(key, [])) > step and np.isfinite(row[key][step])
                    ]
                    item = {
                        "horizon_hours": step + 1,
                        "mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n_seeds": len(values),
                    }
                    entry["metrics"][key].append(item)
        summary[evaluation] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
