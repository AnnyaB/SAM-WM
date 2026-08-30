from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--out", default="artifacts/figures")
    args = parser.parse_args()
    rows = []
    for filename in args.metrics:
        obj = json.loads(Path(filename).read_text(encoding="utf-8"))
        rows.append((obj["dataset"], obj["metrics"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [row[0] for row in rows]
    maes = [row[1]["mae"] for row in rows]
    rmses = [row[1]["rmse"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], maes, width=width, label="MAE")
    ax.bar([i + width / 2 for i in x], rmses, width=width, label="RMSE")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Temperature error (°C)")
    ax.set_title("SAM-WM ID and zero-shot OOD error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "error_summary.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    coverage = [row[1].get("conformal_coverage") for row in rows]
    ax.bar(names, [0.0 if value is None else value for value in coverage])
    ax.axhline(0.9, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Frozen Freiburg 90% conformal interval coverage")
    fig.tight_layout()
    fig.savefig(out / "uncertainty_coverage.png", dpi=220)
    plt.close(fig)

    horizon_rows = [(name, metrics.get("horizon_mae", [])) for name, metrics in rows]
    if any(values for _, values in horizon_rows):
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, values in horizon_rows:
            if values:
                ax.plot(range(1, len(values) + 1), values, marker="o", label=name)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("MAE (°C)")
        ax.set_title("Horizon-wise forecast degradation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "horizon_mae.png", dpi=220)
        plt.close(fig)

    print(out)


if __name__ == "__main__":
    main()
