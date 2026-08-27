from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.action_evidence import build_action_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CANDRA action artifact from source + independent transfer DiD evidence."
    )
    parser.add_argument("--kind", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--transfer-csv", required=True)
    parser.add_argument("--source-provenance", required=True)
    parser.add_argument("--transfer-provenance", required=True)
    parser.add_argument("--reference-coverage", type=float, required=True)
    parser.add_argument("--coverage-tolerance", type=float, default=0.10)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pairs", type=int, default=48)
    parser.add_argument("--support-reference-pairs", type=int, default=168)
    parser.add_argument("--out", default="artifacts/deployment/candra_actions.json")
    args = parser.parse_args()

    result = build_action_evidence(
        kind=args.kind,
        source_csv=Path(args.source_csv),
        transfer_csv=Path(args.transfer_csv),
        source_provenance=args.source_provenance,
        transfer_provenance=args.transfer_provenance,
        reference_coverage_fraction=args.reference_coverage,
        coverage_tolerance=args.coverage_tolerance,
        horizon=args.horizon,
        out=Path(args.out),
        block=args.block,
        samples=args.samples,
        seed=args.seed,
        min_pairs=args.min_pairs,
        support_reference_pairs=args.support_reference_pairs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
