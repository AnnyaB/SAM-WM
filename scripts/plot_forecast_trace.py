from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coolworld.paper_figures import configure_paper_style, plot_forecast_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate only the representative zero-shot forecast trace from the final "
            "machine-readable paper-suite results."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/paper_suite/paper_suite_results.json"),
        help="Final paper_suite_results.json containing the exact saved trace values.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper_suite/figures"),
        help="Output directory for SVG, vector PDF, and 600-dpi PNG.",
    )
    args = parser.parse_args()

    if not args.results.is_file():
        raise SystemExit(
            f"Results file not found: {args.results}. Import the final Kaggle archive first."
        )

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    configure_paper_style()
    plot_forecast_trace(payload, args.out)
    print(f"Forecast trace written to {args.out}")


if __name__ == "__main__":
    main()
