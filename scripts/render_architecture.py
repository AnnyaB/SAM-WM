from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, title, lines, *, fc="#FFFFFF", ec="#202124", lw=1.3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.035 * w, y + h - 0.18 * h, title,
            ha="left", va="top", fontsize=11.5, fontweight="bold")
    ax.text(x + 0.035 * w, y + h - 0.43 * h, "\n".join(lines),
            ha="left", va="top", fontsize=8.6, linespacing=1.35)
    return patch


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.25, color="#3C4043",
        shrinkA=4, shrinkB=4,
    ))


def render(out: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(15.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "SAM-WM · CoolWorld", fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.03, 0.885,
        "Evidence-bounded urban thermal forecasting and resilient-city decision support",
        fontsize=10.5, color="#5F6368", va="top",
    )

    y, h = 0.29, 0.45
    xs = [0.03, 0.225, 0.42, 0.615, 0.81]
    w = 0.155

    box(ax, xs[0], y, w, h, "1 · REAL EVIDENCE", [
        "FortyGuard TCM",
        "65 hourly frames",
        "36-tile San José grid",
        "timestamp + SHA-256 provenance",
    ], fc="#FFF7F3", ec="#D65F3F")

    box(ax, xs[1], y, w, h, "2 · OBSERVE", [
        "3D measured thermal field",
        "48 h consecutive context",
        "deterministic physical",
        "kNN graph",
        "temperature + RH availability",
    ], fc="#F8FAFC", ec="#607D9A")

    box(ax, xs[2], y, w, h, "3 · SAM-WM", [
        "sparse mental map",
        "exchange + wind transport",
        "bounded source + residual",
        "adaptive router + GRU rollout",
        "+1…+6 h + uncertainty",
    ], fc="#FFF3EE", ec="#E76F51", lw=1.6)

    box(ax, xs[3], y, w, h, "4 · PRIORITIZE", [
        "future temperature",
        "hotspot persistence",
        "forecast uncertainty",
        "rank sites for",
        "engineering review",
    ], fc="#F8FAFC", ec="#607D9A")

    box(ax, xs[4], y, w, h, "5 · VALIDATE", [
        "shade / canopy / reflective",
        "measure treated vs control",
        "effect only with evidence",
        "otherwise: abstain",
    ], fc="#F7F7F7", ec="#6B7280")

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, left + w, y + 0.52*h, right, y + 0.52*h)

    ax.text(0.03, 0.16, "Claim boundary:", fontsize=9.5, fontweight="bold", va="center")
    ax.text(
        0.148, 0.16,
        "SAM-WM forecasts and prioritizes; it does not claim a cooling effect until independent intervention evidence exists.",
        fontsize=9.5, va="center", color="#3C4043"
    )

    out.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 600})]:
        fig.savefig(out / f"samwm_system_architecture.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the deterministic SAM-WM system architecture figure.")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    render(args.out)
    print(f"Architecture figure written to {args.out}")


if __name__ == "__main__":
    main()
