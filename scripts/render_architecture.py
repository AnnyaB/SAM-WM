from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#202124"
MUTED = "#5F6368"
SAM = "#E76F51"
SAM_PALE = "#FFF7F3"
BLUE = "#607D9A"
BLUE_PALE = "#F6F9FB"
GRAY = "#6B7280"
GRAY_PALE = "#F8F8F8"
GREEN = "#5D8A72"
GREEN_PALE = "#F5F9F6"
LINE = "#C9CED3"


def rounded(ax, x, y, w, h, *, fc="white", ec=INK, lw=1.1, radius=0.010):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, *, color="#4B5563", lw=1.15, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def stage(ax, x, y, w, h, number, title, lines, *, fc, ec):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.15)
    ax.text(
        x + 0.055 * w,
        y + h - 0.16 * h,
        f"{number}  {title}",
        ha="left",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + 0.055 * w,
        y + h - 0.42 * h,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=7.45,
        linespacing=1.35,
        color=INK,
    )


def small_box(ax, x, y, w, h, title, subtitle, *, ec=LINE):
    rounded(ax, x, y, w, h, fc="white", ec=ec, lw=0.8, radius=0.006)
    ax.text(
        x + w / 2,
        y + 0.62 * h,
        title,
        ha="center",
        va="center",
        fontsize=6.9,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + w / 2,
        y + 0.28 * h,
        subtitle,
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
    )


def model_block(ax, x, y, w, h):
    rounded(ax, x, y, w, h, fc=SAM_PALE, ec=SAM, lw=1.55, radius=0.012)
    ax.text(
        x + 0.035 * w,
        y + h - 0.075 * h,
        "3  SAM-WM",
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + 0.035 * w,
        y + h - 0.17 * h,
        "Sparse Adaptive Mechanism World Model  ·  48 h context  →  recurrent +1…+6 h rollout",
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    # Representation and routing path.
    route_y = y + 0.58 * h
    route_h = 0.18 * h
    left = x + 0.04 * w
    total = 0.92 * w
    gap = 0.025 * w
    box_w = (total - 2 * gap) / 3
    route_specs = [
        ("Sparse physical graph", "deterministic kNN"),
        ("Adaptive mental map", "state-dependent messages"),
        ("Mechanism router", "latent state + time"),
    ]
    for idx, (title, subtitle) in enumerate(route_specs):
        xx = left + idx * (box_w + gap)
        small_box(ax, xx, route_y, box_w, route_h, title, subtitle)
        if idx < 2:
            arrow(
                ax,
                xx + box_w,
                route_y + route_h / 2,
                xx + box_w + gap,
                route_y + route_h / 2,
                color="#858B91",
                lw=0.75,
            )

    ax.text(
        x + 0.04 * w,
        y + 0.505 * h,
        "Typed dynamics operators",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=MUTED,
    )

    # Four typed mechanisms in a 2×2 grid for legibility.
    mech_left = x + 0.04 * w
    mech_top = y + 0.455 * h
    mech_gap_x = 0.018 * w
    mech_gap_y = 0.020 * h
    mech_w = (0.92 * w - mech_gap_x) / 2
    mech_h = 0.105 * h
    mechanisms = [
        (r"$\Delta^{ex}$", "conservative exchange"),
        (r"$\Delta^{wind}$", "upwind transport"),
        (r"$\Delta^{src}$", "bounded source / sink"),
        (r"$\Delta^{res}$", "bounded residual"),
    ]
    for idx, (symbol, label) in enumerate(mechanisms):
        row, col = divmod(idx, 2)
        xx = mech_left + col * (mech_w + mech_gap_x)
        yy = mech_top - row * (mech_h + mech_gap_y) - mech_h
        rounded(
            ax,
            xx,
            yy,
            mech_w,
            mech_h,
            fc="white",
            ec="#D6A48F",
            lw=0.8,
            radius=0.005,
        )
        ax.text(
            xx + 0.08 * mech_w,
            yy + mech_h / 2,
            symbol,
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=SAM,
        )
        ax.text(
            xx + 0.30 * mech_w,
            yy + mech_h / 2,
            label,
            ha="left",
            va="center",
            fontsize=6.2,
            color=INK,
        )

    ax.text(
        x + 0.04 * w,
        y + 0.185 * h,
        r"$T_i^{t+1}=T_i^t+\Delta_i^{ex}+\Delta_i^{wind}+\Delta_i^{src}+\Delta_i^{res}$",
        ha="left",
        va="center",
        fontsize=8.8,
        color=INK,
    )
    ax.text(
        x + 0.04 * w,
        y + 0.095 * h,
        "Uncertainty: learned Laplace scale  →  Freiburg validation calibration  →  frozen OOD calibration",
        ha="left",
        va="center",
        fontsize=6.35,
        color=MUTED,
    )
    ax.text(
        x + 0.04 * w,
        y + 0.035 * h,
        "Structural claim only: exchange / transport are conservative operators; the complete model is not globally energy-conserving.",
        ha="left",
        va="center",
        fontsize=5.95,
        color=MUTED,
    )


def render(out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(17.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.025,
        0.955,
        "SAM-WM · CoolWorld",
        fontsize=18.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    ax.text(
        0.025,
        0.905,
        "From immutable FortyGuard thermal evidence to uncertainty-aware resilient-city engineering review",
        fontsize=9.8,
        ha="left",
        va="top",
        color=MUTED,
    )

    # Evidence-to-forecast path.
    y = 0.50
    stage(
        ax,
        0.025,
        y,
        0.135,
        0.285,
        "1",
        "REAL EVIDENCE",
        [
            "FortyGuard TCM",
            "65 recorded hourly frames",
            "one 36-tile San José grid",
            "timestamp + SHA-256 provenance",
        ],
        fc=BLUE_PALE,
        ec=BLUE,
    )
    stage(
        ax,
        0.182,
        y,
        0.13,
        0.285,
        "2",
        "OBSERVE IN 3D",
        [
            "measured thermal field",
            "48 h consecutive context",
            "temperature + RH availability",
            "deterministic physical graph",
        ],
        fc=BLUE_PALE,
        ec=BLUE,
    )
    model_block(ax, 0.335, 0.405, 0.375, 0.45)
    stage(
        ax,
        0.735,
        y,
        0.105,
        0.285,
        "4",
        "FORECAST",
        ["+1…+6 h field", "temperature", "uncertainty", "research forecast"],
        fc=SAM_PALE,
        ec=SAM,
    )
    stage(
        ax,
        0.865,
        y,
        0.11,
        0.285,
        "5",
        "PRIORITIZE",
        ["persistent heat", "hotspot ranking", "uncertainty shown", "no causal claim"],
        fc=GREEN_PALE,
        ec=GREEN,
    )

    arrow(ax, 0.160, y + 0.1425, 0.182, y + 0.1425)
    arrow(ax, 0.312, y + 0.1425, 0.335, y + 0.1425)
    arrow(ax, 0.710, y + 0.1425, 0.735, y + 0.1425)
    arrow(ax, 0.840, y + 0.1425, 0.865, y + 0.1425)

    # Evidence-gated intervention / validation path.
    bottom_y, bottom_h = 0.105, 0.19
    stage(
        ax,
        0.465,
        bottom_y,
        0.155,
        bottom_h,
        "6",
        "ENGINEERING REVIEW",
        [
            "shade / canopy / reflective surface",
            "choose a physical intervention",
            "forecast guides review only",
        ],
        fc=GRAY_PALE,
        ec=GRAY,
    )
    stage(
        ax,
        0.645,
        bottom_y,
        0.155,
        bottom_h,
        "7",
        "TREATED VS CONTROL",
        [
            "measure after intervention",
            "store independent evidence",
            "compare treated / control",
        ],
        fc=GRAY_PALE,
        ec=GRAY,
    )
    stage(
        ax,
        0.825,
        bottom_y,
        0.15,
        bottom_h,
        "8",
        "VALIDATE / ABSTAIN",
        [
            "report effect only if supported",
            "otherwise preserve abstention",
            "causal gate stays closed",
        ],
        fc=GRAY_PALE,
        ec=GRAY,
    )

    arrow(ax, 0.92, y, 0.555, bottom_y + bottom_h, color=GRAY, lw=1.0, rad=-0.18)
    arrow(ax, 0.620, bottom_y + bottom_h / 2, 0.645, bottom_y + bottom_h / 2, color=GRAY)
    arrow(ax, 0.800, bottom_y + bottom_h / 2, 0.825, bottom_y + bottom_h / 2, color=GRAY)

    # Evidence boundary annotation.
    rounded(ax, 0.025, 0.105, 0.395, 0.19, fc="white", ec="#D5D9DD", lw=0.9)
    ax.text(
        0.043,
        0.258,
        "Evidence boundary",
        fontsize=9.1,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )
    ax.text(
        0.043,
        0.225,
        "observed evidence  ≠  research forecast  ≠  operational certification  ≠  causal intervention evidence",
        fontsize=7.0,
        ha="left",
        va="top",
        color=INK,
    )
    ax.text(
        0.043,
        0.178,
        "SAM-WM forecasts where heat may persist; it does not claim that a tree, shade structure, reflective",
        fontsize=6.85,
        ha="left",
        va="top",
        color=MUTED,
    )
    ax.text(
        0.043,
        0.145,
        "material or other action caused cooling until independent treated / control evidence exists.",
        fontsize=6.85,
        ha="left",
        va="top",
        color=MUTED,
    )

    out.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(
            out / f"samwm_system_architecture.{suffix}",
            bbox_inches="tight",
            pad_inches=0.05,
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render deterministic, evidence-bounded SAM-WM system architecture."
    )
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    render(args.out)
    print(f"Architecture figure written to {args.out}")


if __name__ == "__main__":
    main()
