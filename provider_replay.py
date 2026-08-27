from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.provider import evaluate_provider_replay, recorded_heatmap_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SAM-WM transfer on recorded real FortyGuard TCM evidence."
    )
    parser.add_argument("--checkpoint", default="artifacts/deployment/best.pt")
    parser.add_argument("--calibration", default="artifacts/deployment/calibration.json")
    parser.add_argument("--evaluation", default="artifacts/deployment/evaluation.json")
    parser.add_argument("--evidence", default="artifacts/fortyguard")
    parser.add_argument("--out", default="artifacts/deployment/fortyguard_replay.json")
    parser.add_argument("--limit", type=int, default=240)
    args = parser.parse_args()

    frames = recorded_heatmap_frames(Path(args.evidence), limit=args.limit)
    result = evaluate_provider_replay(
        Path(args.checkpoint),
        Path(args.calibration),
        Path(args.evaluation),
        frames,
        Path(args.out),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
