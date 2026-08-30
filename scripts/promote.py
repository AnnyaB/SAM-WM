from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.promotion import finalize_deployment_bundle, select_deployment_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preselect and promote one frozen SAM-WM deployment checkpoint."
    )
    parser.add_argument("stage", choices=["preselect", "finalize"])
    parser.add_argument("--research-root", default="artifacts/research")
    parser.add_argument("--source-sha", default="artifacts/FROZEN_SOURCE_SHA.txt")
    parser.add_argument("--selection", default="artifacts/DEPLOYMENT_SELECTION.json")
    parser.add_argument("--freeze", default="artifacts/FREEZE_MANIFEST.json")
    parser.add_argument("--eval-root", default="artifacts/eval")
    parser.add_argument("--deployment-root", default="artifacts/deployment")
    args = parser.parse_args()

    if args.stage == "preselect":
        result = select_deployment_seed(
            Path(args.research_root),
            Path(args.source_sha),
            Path(args.selection),
        )
    else:
        result = finalize_deployment_bundle(
            Path(args.selection),
            Path(args.freeze),
            Path(args.eval_root),
            Path(args.deployment_root),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
