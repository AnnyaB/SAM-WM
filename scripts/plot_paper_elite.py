from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SAM = "#E76F51"
BLUE = "#607D9A"
GRAY = "#9AA0A6"
INK = "#202124"
MUTED = "#5F6368"
LIGHT = "#DADCE0"
ABLATION = ["#E76F51", "#D9A441", "#7C9EB2", "#8A7AA5", "#6F9E8B", "#B47C6C"]

DISPLAY = {
    "samwm": "SAM-WM",
    "itransformer": "iTransformer-adapted",
    "timemixer": "TimeMixer-adapted",
    "samwm_no_sigreg": "− SIGReg",
    "samwm_no_exchange": "− exchange",
    "samwm_no_mental_map": "− mental map",
    "samwm_no_residual": "− residual",
    "samwm_no_rh": "− RH",
}

FULL_MODELS = ("samwm", "itransformer", "timemixer")
ABLATIONS = (
    "samwm",
    "samwm_no_sigreg",
    "samwm_no_exchange",
    "samwm_no_mental_map",
    "samwm_no_residual",
    "samwm_no_rh",
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _panel(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _metric(results: dict, model: str, domain: str, key: str) -> tuple[float, float]:
    value = results["summary"][model]["domains"][domain]["metrics"][key]
    return float(value["mean"]), float(value["std"])


def benchmark_overview(results: dict, out: Path) -> None:
    colors = [SAM, BLUE, GRAY]
    labels = [DISPLAY[m] for m in FULL_MODELS]
    domains = ("freiburg", "novisad")
    domain_labels = ("Freiburg ID", "Novi Sad OOD")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.15), gridspec_kw={"wspace": 0.34})
    x = np.arange(2)
    width = 0.22
    for idx, model in enumerate(FULL_MODELS):
        maes = [_metric(results, model, d, "mae")[0] for d in domains]
        mae_sd = [_metric(results, model, d, "mae")[1] for d in domains]
        axes[0].bar(
            x + (idx - 1) * width,
            maes,
            width,
            yerr=mae_sd,
            capsize=3,
            color=colors[idx],
            edgecolor="none",
            label=labels[idx],
        )
    axes[0].set_xticks(x, domain_labels)
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[0].set_title("Cross-city forecast accuracy")
    axes[0].set_ylim(0, 5.0)
    _panel(axes[0], "(a)")
    _clean(axes[0])

    for idx, model in enumerate(FULL_MODELS):
        cov = [100 * _metric(results, model, d, "conformal_coverage")[0] for d in domains]
        cov_sd = [100 * _metric(results, model, d, "conformal_coverage")[1] for d in domains]
        axes[1].errorbar(
            x,
            cov,
            yerr=cov_sd,
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=3,
            color=colors[idx],
            label=labels[idx],
        )
    axes[1].axhline(90, color=MUTED, linewidth=1.1, linestyle="--")
    axes[1].text(0.95, 90.6, "90% nominal", ha="right", color=INK, fontsize=9)
    axes[1].set_xticks(x, domain_labels)
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_title("Frozen source calibration")
    axes[1].set_ylim(40, 94)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _panel(axes[1], "(b)")
    _clean(axes[1])

    _save(fig, out, "benchmark_overview")


def horizon_transfer(results: dict, out: Path) -> None:
    colors = [SAM, BLUE, GRAY]
    labels = [DISPLAY[m] for m in FULL_MODELS]
    domains = (("freiburg", "Freiburg ID"), ("novisad", "Novi Sad OOD"))
    h = np.arange(1, 7)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.28})
    for ax, (domain, title) in zip(axes, domains, strict=True):
        for model, color, label in zip(FULL_MODELS, colors, labels, strict=True):
            rows = results["summary"][model]["domains"][domain]["metrics"]["horizon_mae"]
            mean = np.array([float(r["mean"]) for r in rows])
            sd = np.array([float(r["std"]) for r in rows])
            ax.plot(h, mean, marker="o", markersize=4.5, linewidth=2.0, color=color, label=label)
            ax.fill_between(h, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
        ax.set_xticks(h)
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_ylabel("MAE (°C) ↓")
        ax.set_title(title)
        _clean(ax)
    _panel(axes[0], "(a)")
    _panel(axes[1], "(b)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, out, "horizon_transfer")


def ablation_study(results: dict, out: Path) -> None:
    domains = ("freiburg", "novisad")
    labels = [DISPLAY[m] for m in ABLATIONS]
    x = np.arange(len(ABLATIONS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    domain_styles = zip(domains, (-width / 2, width / 2), (None, "//"), strict=True)
    for j, (domain, shift, hatch) in enumerate(domain_styles):
        vals = [_metric(results, m, domain, "mae")[0] for m in ABLATIONS]
        sds = [_metric(results, m, domain, "mae")[1] for m in ABLATIONS]
        bars = ax.bar(
            x + shift,
            vals,
            width,
            yerr=sds,
            capsize=2.5,
            color=ABLATION,
            alpha=1.0 if j == 0 else 0.55,
            edgecolor=INK if j == 1 else "none",
            linewidth=0.5,
            hatch=hatch,
            label="Freiburg ID" if j == 0 else "Novi Sad OOD",
        )
        if j == 1:
            for bar in bars:
                bar.set_edgecolor("white")
    ax.axhline(
        _metric(results, "samwm", "freiburg", "mae")[0],
        color=MUTED,
        linewidth=0.9,
        linestyle=":",
    )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAE (°C) ↓")
    ax.set_title("SAM-WM component ablations")
    ax.set_ylim(1.15, 1.75)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _clean(ax)
    _save(fig, out, "ablation_study")


def _history_matrix(history_summary: dict, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = history_summary[model]["validation_mae_c"]
    epochs = np.array([int(row["epoch"]) for row in rows], dtype=int)
    mean = np.array([float(row["mean"]) for row in rows], dtype=float)
    std = np.array([float(row["std"]) for row in rows], dtype=float)
    return epochs, mean, std


def learning_dynamics(history_summary: dict, out: Path) -> None:
    colors = [SAM, BLUE, GRAY]
    labels = [DISPLAY[m] for m in FULL_MODELS]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for model, color, label in zip(FULL_MODELS, colors, labels, strict=True):
        e, mean, sd = _history_matrix(history_summary, model)
        ax.plot(e, mean, linewidth=2.0, color=color, label=label)
        ax.fill_between(e, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Freiburg validation MAE (°C) ↓")
    ax.set_title("Source-domain learning dynamics")
    ax.legend(frameon=False)
    _clean(ax)
    _save(fig, out, "learning_dynamics")


def calibration_efficiency(results: dict, out: Path) -> None:
    colors = [SAM, BLUE, GRAY]
    labels = [DISPLAY[m] for m in FULL_MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(FULL_MODELS))
    cov = [100 * _metric(results, m, "novisad", "conformal_coverage")[0] for m in FULL_MODELS]
    cov_sd = [100 * _metric(results, m, "novisad", "conformal_coverage")[1] for m in FULL_MODELS]
    axes[0].bar(x, cov, yerr=cov_sd, capsize=3, color=colors, width=0.62)
    axes[0].axhline(90, color=MUTED, linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x, ["SAM-WM", "iTransformer\nadapted", "TimeMixer\nadapted"])
    axes[0].set_ylabel("Novi Sad coverage (%)")
    axes[0].set_title("Source-frozen uncertainty transfer")
    axes[0].set_ylim(35, 94)
    _panel(axes[0], "(a)")
    _clean(axes[0])

    for model, color, label in zip(FULL_MODELS, colors, labels, strict=True):
        params = _metric(results, model, "freiburg", "parameter_count")[0]
        ood = _metric(results, model, "novisad", "mae")[0]
        axes[1].scatter(params / 1000.0, ood, s=62, color=color, label=label, zorder=3)
    axes[1].set_xlabel("Trainable parameters (thousands)")
    axes[1].set_ylabel("Novi Sad MAE (°C) ↓")
    axes[1].set_title("Compactness vs zero-shot error")
    axes[1].legend(frameon=False, loc="upper right")
    _panel(axes[1], "(b)")
    _clean(axes[1])
    _save(fig, out, "calibration_efficiency")


def frozen_three_domain(frozen: dict, out: Path) -> None:
    keys = ("freiburg_heldout", "novisad_heldout", "fairurbtemp_heldout")
    labels = ("Freiburg ID", "Novi Sad OOD", "Turku OOD")
    x = np.arange(3)
    mae = [float(frozen[k]["metrics"]["mae"]["mean"]) for k in keys]
    mae_sd = [float(frozen[k]["metrics"]["mae"]["std"]) for k in keys]
    cov = [100 * float(frozen[k]["metrics"]["conformal_coverage"]["mean"]) for k in keys]
    cov_sd = [100 * float(frozen[k]["metrics"]["conformal_coverage"]["std"]) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), gridspec_kw={"wspace": 0.32})
    axes[0].bar(x, mae, yerr=mae_sd, capsize=3, color=[SAM, "#F08A6B", "#F3A489"], width=0.62)
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[0].set_title("Frozen SAM-WM across three cities")
    axes[0].set_ylim(0, 1.8)
    _panel(axes[0], "(a)")
    _clean(axes[0])

    axes[1].errorbar(x, cov, yerr=cov_sd, color=SAM, marker="o", linewidth=2.0, capsize=3)
    axes[1].axhline(90, color=MUTED, linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_title("Frozen source calibration")
    axes[1].set_ylim(82, 93)
    _panel(axes[1], "(b)")
    _clean(axes[1])
    _save(fig, out, "frozen_three_domain")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evidence-only SAM-WM paper/README figures."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/paper_suite/paper_suite_results.json"),
    )
    parser.add_argument(
        "--learning-history",
        type=Path,
        default=Path("results/paper_suite/learning_history_summary.json"),
    )
    parser.add_argument("--frozen-summary", type=Path, default=Path("artifacts/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("results/paper_suite/figures"))
    args = parser.parse_args()

    _style()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    if "summary" not in results:
        base = args.results.parent
        results["summary"] = {
            model: json.loads((base / path).read_text(encoding="utf-8"))
            for model, path in results["summary_files"].items()
        }
    history_summary = json.loads(args.learning_history.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_summary.read_text(encoding="utf-8"))

    benchmark_overview(results, args.out)
    horizon_transfer(results, args.out)
    ablation_study(results, args.out)
    learning_dynamics(history_summary, args.out)
    calibration_efficiency(results, args.out)
    frozen_three_domain(frozen, args.out)
    print(f"Elite evidence figures written to {args.out}")


if __name__ == "__main__":
    main()
