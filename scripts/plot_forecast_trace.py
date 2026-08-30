from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SAM_RED = "#E76F51"
BASELINE_BLUE = "#607D9A"
BASELINE_GREY = "#9AA0A6"
INK = "#202124"

MODEL_COLORS = {
    "samwm": SAM_RED,
    "itransformer": BASELINE_BLUE,
    "timemixer": BASELINE_GREY,
}

MODEL_LABELS = {
    "samwm": "SAM-WM",
    "itransformer": "iTransformer-adapted",
    "timemixer": "TimeMixer-adapted",
}

DOMAIN_LABELS = {
    "freiburg": "Freiburg ID",
    "novisad": "Novi Sad OOD",
    "turku": "Turku OOD",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
        }
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(out / f"forecast_trace.{suffix}", **kwargs)
    plt.close(fig)


def choose_domain(payload: dict[str, Any], models: list[str]) -> str:
    domains = payload["summary"][models[0]]["domains"]
    if "turku" in domains:
        return "turku"
    if "novisad" in domains:
        return "novisad"
    raise KeyError("Expected a Novi Sad or Turku OOD domain in final results")


def render(payload: dict[str, Any], out: Path) -> None:
    models = [
        name
        for name in ("samwm", "itransformer", "timemixer")
        if name in payload["models"]
    ]
    if not models:
        raise KeyError("No expected forecast models found in final results")

    domain = choose_domain(payload, models)
    x = np.arange(1, 7)

    first = payload["raw"][models[0]]["domains"][domain][0]["trace"]
    target = np.asarray(first["target_mean_c"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.1, 2.55))
    ax.plot(
        x,
        target,
        color=INK,
        marker="o",
        linewidth=1.6,
        label="Observed",
        zorder=4,
    )

    for model in models:
        trace = payload["raw"][model]["domains"][domain][0]["trace"]
        prediction = np.asarray(trace["prediction_mean_c"], dtype=float)
        ax.plot(
            x,
            prediction,
            color=MODEL_COLORS[model],
            marker="o",
            label=MODEL_LABELS[model],
            zorder=3 if model == "samwm" else 2,
        )

    ax.set_xticks(x)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Spatial-mean temperature (°C)")
    ax.set_title(f"Representative zero-shot rollout · {DOMAIN_LABELS[domain]}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.8,
        labelspacing=0.65,
    )
    fig.subplots_adjust(right=0.72)
    save_figure(fig, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the representative OOD forecast trace directly from the final "
            "machine-readable paper-suite results. This script intentionally has no "
            "dependency on the SAM-WM package or PyTorch."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/paper_suite/paper_suite_results.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper_suite/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise SystemExit(
            f"Results file not found: {args.results}. Import the final Kaggle archive first."
        )

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    configure_style()
    render(payload, args.out)
    print(f"Forecast trace written to {args.out}")


if __name__ == "__main__":
    main()
