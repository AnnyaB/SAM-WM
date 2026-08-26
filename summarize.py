from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="artifacts/eval")
    p.add_argument("--out", default="artifacts/summary.json")
    a = p.parse_args()
    root = Path(a.root)
    records = {}
    for path in root.glob("seed_*/*_metrics.json"):
        obj = json.loads(path.read_text())
        records.setdefault(obj["dataset"], []).append(obj["metrics"])
    if not records:
        raise SystemExit(f"no metrics under {root}/seed_*")
    summary = {}
    for dataset, rows in sorted(records.items()):
        summary[dataset] = {}
        for key in [
            "mae",
            "rmse",
            "bias",
            "p95_abs_error",
            "conformal_90_coverage",
            "mean_surprise",
        ]:
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                summary[dataset][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
