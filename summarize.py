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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/eval")
    parser.add_argument("--out", default="artifacts/summary.json")
    args = parser.parse_args()
    root = Path(args.root)
    records: dict[str, list[dict]] = {}
    for path in root.glob("seed_*/*_metrics.json"):
        obj = json.loads(path.read_text(encoding="utf-8"))
        records.setdefault(obj["dataset"], []).append(obj["metrics"])
    if not records:
        raise SystemExit(f"no metrics under {root}/seed_*")

    summary = {}
    for dataset, rows in sorted(records.items()):
        summary[dataset] = {}
        for key in METRICS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if values:
                summary[dataset][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n_seeds": len(values),
                }
        horizon_count = max((len(row.get("horizon_mae", [])) for row in rows), default=0)
        if horizon_count:
            summary[dataset]["horizon_mae"] = []
            summary[dataset]["horizon_rmse"] = []
            for step in range(horizon_count):
                for key in ("horizon_mae", "horizon_rmse"):
                    values = [
                        float(row[key][step])
                        for row in rows
                        if len(row.get(key, [])) > step and np.isfinite(row[key][step])
                    ]
                    summary[dataset][key].append(
                        {
                            "horizon_hours": step + 1,
                            "mean": float(np.mean(values)) if values else None,
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "n_seeds": len(values),
                        }
                    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
