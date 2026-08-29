from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.paper_figures import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM-WM paper figures from the matched Kaggle benchmark suite."
    )
    parser.add_argument(
        "--results",
        default="artifacts/paper_suite/paper_suite_results.json",
        help="Machine-readable output produced by coolworld.paper_suite.",
    )
    parser.add_argument(
        "--root",
        default="artifacts/paper_suite",
        help="Paper-suite root containing per-seed training histories.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/paper_suite/figures",
        help="Output directory for PDF, SVG and 600-dpi PNG figures.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        raise SystemExit(
            f"paper-suite results not found: {results_path}. "
            "Run `python -m coolworld.paper_suite ...` first."
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    out = generate_all(payload, root=args.root, out=args.out)
    print(f"Publication figures written to {out}")


if __name__ == "__main__":
    main()
