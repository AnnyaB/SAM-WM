from __future__ import annotations

import argparse
import json

from coolworld.ml.split import split_sequence_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a real SAM-WM sequence bundle into purged chronological "
            "development, calibration, and final-test partitions."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    result = split_sequence_bundle(
        args.dataset,
        args.manifest,
        args.output_dir,
        calibration_fraction=args.calibration_fraction,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
