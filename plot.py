from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+")
    p.add_argument("--out", default="artifacts/figures")
    a = p.parse_args()
    rows = []
    for f in a.metrics:
        obj = json.loads(Path(f).read_text())
        rows.append((obj["dataset"], obj["metrics"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [r[0] for r in rows]
    maes = [r[1]["mae"] for r in rows]
    rmses = [r[1]["rmse"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    w = 0.35
    ax.bar([i - w / 2 for i in x], maes, width=w, label="MAE")
    ax.bar([i + w / 2 for i in x], rmses, width=w, label="RMSE")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Temperature error (°C)")
    ax.set_title("SAM-WM ID and zero-shot OOD error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_summary.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    coverage = [r[1].get("conformal_90_coverage") for r in rows]
    ax.bar(names, [0 if v is None else v for v in coverage])
    ax.axhline(0.9, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Frozen Freiburg 90% conformal interval coverage")
    fig.tight_layout()
    fig.savefig(out / "uncertainty_coverage.png", dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
