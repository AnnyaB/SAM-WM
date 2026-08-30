from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .paper_models import baseline_display_name

# Restrained publication palette inspired by the visual discipline used in LeWM figures.
SAM_RED = "#E76F51"
SAM_RED_LIGHT = "#F2A38E"
BASELINE_BLUE = "#607D9A"
BASELINE_GREY = "#9AA0A6"
INK = "#202124"
MUTED = "#6B6F76"
LIGHT = "#D8DADF"
ABLATION = ("#F4A582", "#D6604D", "#F7B267", "#B56576", "#C9ADA7")

MODEL_COLORS = {
    "samwm": SAM_RED,
    "itransformer": BASELINE_BLUE,
    "timemixer": BASELINE_GREY,
    "samwm_no_sigreg": ABLATION[0],
    "samwm_no_exchange": ABLATION[1],
    "samwm_no_mental_map": ABLATION[2],
    "samwm_no_residual": ABLATION[3],
    "samwm_no_rh": ABLATION[4],
}

DOMAIN_LABEL = {"freiburg": "Freiburg ID", "novisad": "Novi Sad OOD", "turku": "Turku OOD"}


def configure_paper_style() -> None:
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


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10)


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(out / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def _metric(payload: dict, model: str, domain: str, key: str) -> tuple[float, float]:
    value = payload["summary"][model]["domains"][domain]["metrics"][key]
    return float(value["mean"]), float(value["std"])


def _horizon(payload: dict, model: str, domain: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = payload["summary"][model]["domains"][domain]["metrics"][key]
    return (
        np.asarray([float(row["mean"]) for row in rows]),
        np.asarray([float(row["std"]) for row in rows]),
    )


def plot_main_results(payload: dict, out: Path) -> None:
    models = [name for name in ("samwm", "itransformer", "timemixer") if name in payload["models"]]
    domains = [
        domain
        for domain in ("freiburg", "novisad", "turku")
        if domain in payload["summary"][models[0]]["domains"]
    ]
    fig, axes = plt.subplots(1, len(domains), figsize=(7.05, 2.20), sharey=True)
    if len(domains) == 1:
        axes = np.asarray([axes])
    x = np.arange(1, 7)
    for idx, (ax, domain) in enumerate(zip(axes, domains, strict=True)):
        for model in models:
            mean, std = _horizon(payload, model, domain, "horizon_mae")
            color = MODEL_COLORS[model]
            ax.plot(
                x,
                mean,
                color=color,
                marker="o",
                label=baseline_display_name(model),
                zorder=3 if model == "samwm" else 2,
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
        ax.set_title(DOMAIN_LABEL[domain])
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_xticks(x)
        _clean(ax)
        _panel(ax, f"({chr(97 + idx)})")
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02))
    fig.subplots_adjust(wspace=0.28)
    _save(fig, out, "main_horizon_results")


def plot_domain_summary(payload: dict, out: Path) -> None:
    models = [name for name in ("samwm", "itransformer", "timemixer") if name in payload["models"]]
    domains = [
        domain
        for domain in ("freiburg", "novisad", "turku")
        if domain in payload["summary"][models[0]]["domains"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.55, 2.35))
    width = 0.22
    centers = np.arange(len(domains))
    for offset, model in enumerate(models):
        means, stds = zip(
            *[_metric(payload, model, domain, "mae") for domain in domains], strict=True
        )
        pos = centers + (offset - (len(models) - 1) / 2) * width
        axes[0].bar(
            pos,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            label=baseline_display_name(model),
        )
    axes[0].set_xticks(centers, [DOMAIN_LABEL[d] for d in domains], rotation=12, ha="right")
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[0].set_title("Cross-city forecast accuracy")
    _clean(axes[0])
    _panel(axes[0], "(a)")

    for model in models:
        coverage = [100.0 * _metric(payload, model, d, "conformal_coverage")[0] for d in domains]
        std = [100.0 * _metric(payload, model, d, "conformal_coverage")[1] for d in domains]
        axes[1].errorbar(
            centers,
            coverage,
            yerr=std,
            color=MODEL_COLORS[model],
            marker="o",
            capsize=2,
            label=baseline_display_name(model),
        )
    axes[1].axhline(90.0, color=INK, linestyle="--", linewidth=0.9, alpha=0.8)
    axes[1].text(
        len(domains) - 1 + 0.03,
        90.15,
        "90% nominal",
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    axes[1].set_xticks(centers, [DOMAIN_LABEL[d] for d in domains], rotation=12, ha="right")
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_title("Frozen source calibration")
    _clean(axes[1])
    _panel(axes[1], "(b)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.02))
    fig.subplots_adjust(wspace=0.35)
    _save(fig, out, "forecast_and_calibration")


def plot_ablation(payload: dict, out: Path) -> None:
    variants = [
        name
        for name in (
            "samwm",
            "samwm_no_sigreg",
            "samwm_no_exchange",
            "samwm_no_mental_map",
            "samwm_no_residual",
            "samwm_no_rh",
        )
        if name in payload["models"]
    ]
    domains = [
        domain
        for domain in ("freiburg", "novisad", "turku")
        if domain in payload["summary"][variants[0]]["domains"]
    ]
    fig, axes = plt.subplots(1, len(domains), figsize=(7.05, 2.55), sharey=True)
    if len(domains) == 1:
        axes = np.asarray([axes])
    labels = [
        "Full",
        "− SIGReg",
        "− exchange",
        "− mental map",
        "− residual",
        "− RH",
    ][: len(variants)]
    for idx, (ax, domain) in enumerate(zip(axes, domains, strict=True)):
        means, stds = zip(
            *[_metric(payload, model, domain, "mae") for model in variants], strict=True
        )
        colors = [MODEL_COLORS[m] for m in variants]
        ax.bar(
            np.arange(len(variants)),
            means,
            yerr=stds,
            capsize=2,
            color=colors,
            edgecolor="white",
        )
        ax.set_xticks(np.arange(len(variants)), labels, rotation=55, ha="right")
        ax.set_title(DOMAIN_LABEL[domain])
        _clean(ax)
        _panel(ax, f"({chr(97 + idx)})")
    axes[0].set_ylabel("MAE (°C) ↓")
    fig.subplots_adjust(wspace=0.20, bottom=0.34)
    _save(fig, out, "samwm_ablations")


def plot_efficiency(payload: dict, out: Path) -> None:
    models = [name for name in ("samwm", "itransformer", "timemixer") if name in payload["models"]]
    domain = "freiburg"
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35))
    for model in models:
        mae, mae_std = _metric(payload, model, domain, "mae")
        params, _ = _metric(payload, model, domain, "parameter_count")
        latency, latency_std = _metric(payload, model, domain, "latency_ms_per_window")
        axes[0].errorbar(
            params / 1e6,
            mae,
            yerr=mae_std,
            marker="o",
            color=MODEL_COLORS[model],
            capsize=2,
            label=baseline_display_name(model),
        )
        axes[1].errorbar(
            latency,
            mae,
            xerr=latency_std,
            yerr=mae_std,
            marker="o",
            color=MODEL_COLORS[model],
            capsize=2,
        )
    axes[0].set_xlabel("Trainable parameters (M) ↓")
    axes[0].set_ylabel("Freiburg MAE (°C) ↓")
    axes[0].set_title("Parameter efficiency")
    axes[0].set_xscale("log")
    _clean(axes[0])
    _panel(axes[0], "(a)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    axes[1].set_xlabel("Inference latency / window (ms) ↓")
    axes[1].set_ylabel("Freiburg MAE (°C) ↓")
    axes[1].set_title("Accuracy–latency trade-off")
    _clean(axes[1])
    _panel(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.37)
    _save(fig, out, "efficiency")


def _histories(root: Path, model: str, seeds: list[int]) -> list[list[dict[str, Any]]]:
    histories: list[list[dict[str, Any]]] = []
    for seed in seeds:
        path = root / "runs" / model / f"seed_{seed}" / "history.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        histories.append(payload["history"])
    return histories


def plot_learning_curves(payload: dict, root: Path, out: Path) -> None:
    models = [name for name in ("samwm", "itransformer", "timemixer") if name in payload["models"]]
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for model in models:
        histories = _histories(root, model, payload["seeds"])
        if not histories:
            continue
        max_common = min(len(history) for history in histories)
        values = np.asarray(
            [[float(row["val_mae_c"]) for row in history[:max_common]] for history in histories],
            dtype=float,
        )
        epochs = np.arange(1, max_common + 1)
        mean = values.mean(0)
        std = values.std(0)
        color = MODEL_COLORS[model]
        ax.plot(epochs, mean, color=color, label=baseline_display_name(model))
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (°C) ↓")
    ax.set_title("Learning dynamics")
    _clean(ax)
    ax.legend(loc="upper right")
    _save(fig, out, "learning_curves")


def plot_forecast_trace(payload: dict, out: Path) -> None:
    models = [name for name in ("samwm", "itransformer", "timemixer") if name in payload["models"]]
    domain = "turku" if "turku" in payload["summary"][models[0]]["domains"] else "novisad"
    x = np.arange(1, 7)

    # Keep the data panel compact and reserve a clean right-hand column for the legend.
    # This avoids placing labels on top of the trajectories while preserving the same
    # underlying trace values and publication palette.
    fig, ax = plt.subplots(figsize=(5.15, 2.45))
    first = payload["raw"][models[0]]["domains"][domain][0]["trace"]
    target = np.asarray(first["target_mean_c"], dtype=float)
    ax.plot(x, target, color=INK, marker="o", linewidth=1.6, label="Observed")
    for model in models:
        trace = payload["raw"][model]["domains"][domain][0]["trace"]
        pred = np.asarray(trace["prediction_mean_c"], dtype=float)
        ax.plot(x, pred, color=MODEL_COLORS[model], marker="o", label=baseline_display_name(model))

    ax.set_xticks(x)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Spatial-mean temperature (°C)")
    ax.set_title(f"Representative zero-shot rollout · {DOMAIN_LABEL[domain]}")
    _clean(ax)

    # Publication-style legend: outside the plotting rectangle, vertically aligned,
    # frameless, and ordered exactly as the plotted trajectories.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.8,
        labelspacing=0.65,
    )
    fig.subplots_adjust(right=0.70)
    _save(fig, out, "forecast_trace")


def generate_all(payload: dict, *, root: str | Path, out: str | Path | None = None) -> Path:
    configure_paper_style()
    root = Path(root)
    out = Path(out) if out is not None else root / "figures"
    plot_main_results(payload, out)
    plot_domain_summary(payload, out)
    plot_ablation(payload, out)
    plot_efficiency(payload, out)
    plot_learning_curves(payload, root, out)
    plot_forecast_trace(payload, out)
    return out