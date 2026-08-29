from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SAM = "#E76F51"
BLUE = "#607D9A"
GREY = "#9AA0A6"
INK = "#202124"
PALETTE = {
    "samwm": SAM,
    "itransformer": BLUE,
    "timemixer": GREY,
    "samwm_no_sigreg": "#8A739D",
    "samwm_no_exchange": "#B28A4A",
    "samwm_no_mental_map": "#5E8D8D",
    "samwm_no_residual": "#7D7D7D",
    "samwm_no_rh": "#A5765A",
}
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
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3.5)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def benchmark_overview(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.34})
    x = np.arange(2)
    width = 0.23
    for j, model in enumerate(FULL_MODELS):
        values, errors = [], []
        for domain in ("freiburg", "novisad"):
            metric = results["summary"][model]["domains"][domain]["metrics"]["mae"]
            values.append(metric["mean"])
            errors.append(metric["std"])
        axes[0].bar(
            x + (j - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            color=PALETTE[model],
            edgecolor="none",
            label=DISPLAY[model],
        )
    axes[0].set_xticks(x, ["Freiburg ID", "Novi Sad OOD"])
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[0].set_title("(a) Cross-city forecast accuracy", loc="left", fontweight="semibold")
    axes[0].set_ylim(0, 5.1)
    _clean(axes[0])

    for model in FULL_MODELS:
        values, errors = [], []
        for domain in ("freiburg", "novisad"):
            metric = results["summary"][model]["domains"][domain]["metrics"]["conformal_coverage"]
            values.append(100 * metric["mean"])
            errors.append(100 * metric["std"])
        axes[1].errorbar(
            x,
            values,
            yerr=errors,
            marker="o",
            ms=5,
            lw=2,
            capsize=3,
            color=PALETTE[model],
            label=DISPLAY[model],
        )
    axes[1].axhline(90, color=INK, lw=1, ls="--", alpha=0.75)
    axes[1].text(0.70, 90.7, "90% nominal", ha="center", fontsize=9)
    axes[1].set_xticks(x, ["Freiburg ID", "Novi Sad OOD"])
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_title("(b) Frozen source calibration", loc="left", fontweight="semibold")
    axes[1].set_ylim(40, 94)
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.78))
    _clean(axes[1])
    _save(fig, out, "benchmark_overview")


def horizon_transfer(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, domain, title in zip(
        axes,
        ("freiburg", "novisad"),
        ("Freiburg held-out ID", "Novi Sad zero-shot OOD"),
        strict=True,
    ):
        for model in FULL_MODELS:
            rows = results["summary"][model]["domains"][domain]["metrics"]["horizon_mae"]
            h = np.asarray([row["horizon_hours"] for row in rows])
            mean = np.asarray([row["mean"] for row in rows])
            std = np.asarray([row["std"] for row in rows])
            ax.plot(h, mean, lw=2.1, marker="o", ms=3.8, color=PALETTE[model], label=DISPLAY[model])
            ax.fill_between(h, mean - std, mean + std, color=PALETTE[model], alpha=0.13, linewidth=0)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_title(title, fontweight="semibold")
        _clean(ax)
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Horizon-wise error under source-only training", y=1.02, fontsize=14, fontweight="semibold")
    _save(fig, out, "horizon_transfer")


def ablation_study(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.30})
    y = np.arange(len(ABLATIONS))
    for ax, domain, title in zip(
        axes,
        ("freiburg", "novisad"),
        ("Freiburg ID", "Novi Sad OOD"),
        strict=True,
    ):
        values, errors = [], []
        for model in ABLATIONS:
            metric = results["summary"][model]["domains"][domain]["metrics"]["mae"]
            values.append(metric["mean"])
            errors.append(metric["std"])
        ax.barh(y, values, xerr=errors, capsize=2.5, color=[PALETTE[m] for m in ABLATIONS], edgecolor="none")
        ax.set_yticks(y, [DISPLAY[m] for m in ABLATIONS])
        ax.invert_yaxis()
        ax.set_xlabel("MAE (°C) ↓")
        ax.set_title(title, fontweight="semibold")
        for i, value in enumerate(values):
            ax.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=8.5)
        _clean(ax)
    fig.suptitle("SAM-WM ablation study: mechanisms and modality", y=1.02, fontsize=14, fontweight="semibold")
    _save(fig, out, "ablation_study")


def learning_dynamics(histories: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"wspace": 0.27})
    for ax, model in zip(axes, FULL_MODELS, strict=True):
        curves = []
        for seed_data in histories[model].values():
            rows = seed_data if isinstance(seed_data, list) else seed_data.get("history", seed_data.get("epochs", []))
            values = []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        value = row.get("val_mae", row.get("validation_mae", row.get("val_mae_c")))
                        if value is not None:
                            values.append(float(value))
            if not values and isinstance(seed_data, dict):
                for key in ("val_mae", "validation_mae", "val_mae_c"):
                    if key in seed_data:
                        values = [float(v) for v in seed_data[key]]
                        break
            curves.append(values)
        length = max(len(c) for c in curves)
        matrix = np.full((len(curves), length), np.nan)
        for i, curve in enumerate(curves):
            matrix[i, : len(curve)] = curve
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        epoch = np.arange(1, length + 1)
        ax.plot(epoch, mean, color=PALETTE[model], lw=2.1)
        ax.fill_between(epoch, mean - std, mean + std, color=PALETTE[model], alpha=0.15, linewidth=0)
        ax.set_title(DISPLAY[model], fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Freiburg validation MAE (°C) ↓")
        _clean(ax)
    fig.suptitle("Actual five-seed validation learning dynamics", y=1.03, fontsize=14, fontweight="semibold")
    _save(fig, out, "learning_dynamics")


def calibration_efficiency(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.33})
    for model in FULL_MODELS:
        fr = results["summary"][model]["domains"]["freiburg"]["metrics"]["conformal_coverage"]
        nv = results["summary"][model]["domains"]["novisad"]["metrics"]["conformal_coverage"]
        axes[0].plot([0, 1], [100 * fr["mean"], 100 * nv["mean"]], marker="o", lw=2, color=PALETTE[model], label=DISPLAY[model])
        axes[0].errorbar([0, 1], [100 * fr["mean"], 100 * nv["mean"]], yerr=[100 * fr["std"], 100 * nv["std"]], fmt="none", capsize=3, color=PALETTE[model])
    axes[0].axhline(90, color=INK, lw=1, ls="--", alpha=0.75)
    axes[0].set_xticks([0, 1], ["Freiburg ID", "Novi Sad OOD"])
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title("(a) Frozen source calibration", loc="left", fontweight="semibold")
    _clean(axes[0])

    for model in FULL_MODELS:
        metrics = results["summary"][model]["domains"]["novisad"]["metrics"]
        params = metrics["parameter_count"]["mean"] / 1000
        mae = metrics["mae"]["mean"]
        axes[1].scatter(params, mae, s=65, color=PALETTE[model], zorder=3)
        axes[1].annotate(DISPLAY[model].replace("-adapted", ""), (params, mae), xytext=(5, 5), textcoords="offset points", fontsize=8.3)
    axes[1].set_xlabel("Trainable parameters (thousands)")
    axes[1].set_ylabel("Novi Sad MAE (°C) ↓")
    axes[1].set_title("(b) Compactness vs OOD error", loc="left", fontweight="semibold")
    _clean(axes[1])
    _save(fig, out, "calibration_efficiency")


def frozen_three_domain(summary: dict, out: Path) -> None:
    domains = (
        ("freiburg_heldout", "Freiburg\nID", INK),
        ("novisad_heldout", "Novi Sad\nOOD-1", BLUE),
        ("fairurbtemp_heldout", "Turku\nOOD-2", SAM),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.28})
    values = [summary[key]["metrics"]["mae"]["mean"] for key, _, _ in domains]
    errors = [summary[key]["metrics"]["mae"]["std"] for key, _, _ in domains]
    x = np.arange(len(domains))
    axes[0].bar(x, values, yerr=errors, capsize=3, color=[color for _, _, color in domains], edgecolor="none")
    axes[0].set_xticks(x, [label for _, label, _ in domains])
    axes[0].set_ylabel("MAE (°C) ↓")
    axes[0].set_title("(a) Frozen full-model transfer", loc="left", fontweight="semibold")
    axes[0].set_ylim(0, 1.8)
    _clean(axes[0])
    for key, label, color in domains:
        rows = summary[key]["metrics"]["horizon_mae"]
        h = [r["horizon_hours"] for r in rows]
        mean = [r["mean"] for r in rows]
        axes[1].plot(h, mean, marker="o", ms=3.5, lw=2, color=color, label=label.replace("\n", " "))
    axes[1].set_xticks(range(1, 7))
    axes[1].set_xlabel("Forecast horizon (h)")
    axes[1].set_ylabel("MAE (°C) ↓")
    axes[1].set_title("(b) +1…+6 h horizon", loc="left", fontweight="semibold")
    axes[1].legend(frameon=False, fontsize=8.3)
    _clean(axes[1])
    _save(fig, out, "frozen_three_domain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/paper_suite/paper_suite_results.json"))
    parser.add_argument("--histories", type=Path, default=Path("results/paper_suite/training_histories.json"))
    parser.add_argument("--frozen-summary", type=Path, default=Path("artifacts/summary.json"))
    parser.add_argument("--out", type=Path, default=Path("results/paper_suite/figures"))
    args = parser.parse_args()
    _style()
    results = json.loads(args.results.read_text())
    histories = json.loads(args.histories.read_text())
    frozen = json.loads(args.frozen_summary.read_text())
    benchmark_overview(results, args.out)
    horizon_transfer(results, args.out)
    ablation_study(results, args.out)
    learning_dynamics(histories, args.out)
    calibration_efficiency(results, args.out)
    frozen_three_domain(frozen, args.out)
    print(f"Publication figures written to {args.out}")


if __name__ == "__main__":
    main()
