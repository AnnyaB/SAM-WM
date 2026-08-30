from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.paper_figures import generate_all
from coolworld.paper_suite import PAPER_MODELS, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matched SAM-WM paper benchmark and generate publication figures."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--out", default="artifacts/paper_suite")
    parser.add_argument("--mode", choices=["paper", "deadline"], default="paper")
    parser.add_argument("--models", nargs="*", default=None, choices=PAPER_MODELS)
    parser.add_argument("--novisad-root", default="data/novisad")
    parser.add_argument("--fairurb-root", default=None)
    parser.add_argument("--fairurb-city", default="Turku")
    parser.add_argument("--skip-fairurb", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    payload = run(args)
    out = Path(args.out)
    figure_dir = generate_all(payload, root=out, out=out / "figures")
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol": payload["protocol"],
                "mode": payload["mode"],
                "models": payload["models"],
                "results": str(out / "paper_suite_results.json"),
                "figures": str(figure_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
