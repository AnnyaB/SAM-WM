from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ORDER = (
    ("freiburg_heldout", "Freiburg ID"),
    ("novisad_heldout", "Novi Sad OOD"),
    ("fairurbtemp_heldout", "Turku OOD"),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(width: float, height: float):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.14, 0.17, 0.82, 0.76])
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def plot_horizon_mae(summary: dict, out: Path) -> None:
    fig, ax = _figure(7.2, 4.4)
    for key, label in ORDER:
        points = summary[key]["metrics"]["horizon_mae"]
        x = [int(item["horizon_hours"]) for item in points]
        y = [float(item["mean"]) for item in points]
        err = [float(item["std"]) for item in points]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=2.5, linewidth=1.8, label=label)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("SAM-WM short-horizon transfer")
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "horizon_mae.svg")


def plot_transfer_mae(summary: dict, out: Path) -> None:
    labels, means, stds = [], [], []
    for key, label in ORDER:
        labels.append(label)
        metric = summary[key]["metrics"]["mae"]
        means.append(float(metric["mean"]))
        stds.append(float(metric["std"]))
    fig, ax = _figure(6.4, 4.2)
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=12, ha="right")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Five-seed SAM-WM evaluation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(means) + 0.35)
    for idx, value in enumerate(means):
        ax.text(idx, value + stds[idx] + 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    _save(fig, out / "transfer_mae.svg")


def plot_coverage(summary: dict, out: Path) -> None:
    labels, means, stds = [], [], []
    for key, label in ORDER:
        labels.append(label)
        metric = summary[key]["metrics"]["conformal_coverage"]
        means.append(100.0 * float(metric["mean"]))
        stds.append(100.0 * float(metric["std"]))
    fig, ax = _figure(6.4, 4.2)
    ax.bar(range(len(labels)), means, yerr=stds, capsize=3)
    ax.axhline(90.0, linestyle="--", linewidth=1.2, label="90% nominal")
    ax.set_xticks(range(len(labels)), labels, rotation=12, ha="right")
    ax.set_ylabel("Empirical conformal coverage (%)")
    ax.set_title("Frozen calibration transferred without OOD recalibration")
    ax.set_ylim(80, 94)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, value in enumerate(means):
        ax.text(idx, value + stds[idx] + 0.35, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    _save(fig, out / "coverage.svg")


def plot_provider_gate(replay: dict, out: Path) -> None:
    coverage = 100.0 * float(replay["conformal_coverage"])
    threshold = 100.0 * float(replay["minimum_required_coverage"])
    fig, ax = _figure(6.4, 3.8)
    ax.bar([0], [coverage], width=0.52)
    ax.axhline(threshold, linestyle="--", linewidth=1.3, label=f"Frozen {threshold:.1f}% gate")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([0], ["FortyGuard replay"])
    ax.set_ylabel("Interval coverage (%)")
    ax.set_title("Operational transfer gate — frozen checkpoint")
    margin = max(0.8, abs(threshold - coverage) * 8)
    ax.set_ylim(min(coverage, threshold) - margin, max(coverage, threshold) + margin)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0, coverage + 0.05, f"{coverage:.4f}%", ha="center", va="bottom", fontsize=10)
    _save(fig, out / "provider_replay_gate.svg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate SAM-WM publication figures from tracked machine-readable evidence."
    )
    parser.add_argument("--summary", default="artifacts/summary.json")
    parser.add_argument("--replay", default="artifacts/deployment/fortyguard_replay.json")
    parser.add_argument("--out", default="assets/figures")
    args = parser.parse_args()

    plt.rcParams["svg.fonttype"] = "none"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = _load(Path(args.summary))
    replay = _load(Path(args.replay))
    plot_horizon_mae(summary, out)
    plot_transfer_mae(summary, out)
    plot_coverage(summary, out)
    plot_provider_gate(replay, out)


if __name__ == "__main__":
    main()
