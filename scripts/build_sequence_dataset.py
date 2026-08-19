from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from coolworld.research.etl import build_sequence_bundle, heatmap_evidence_to_table


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build real-only SAM-WM training sequences from recorded FortyGuard evidence."
    )
    p.add_argument("--evidence-root", default="evidence")
    p.add_argument("--sequence-len", type=int, default=18)
    p.add_argument("--intervention-log", help="Optional real intervention action log Parquet/CSV")
    p.add_argument("--output", default="data/processed/urban_thermal_sequences.npz")
    p.add_argument("--manifest", default="data/processed/urban_thermal_sequences.manifest.json")
    a = p.parse_args()
    table = heatmap_evidence_to_table(a.evidence_root)
    intervention = None
    if a.intervention_log:
        path = Path(a.intervention_log)
        intervention = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = build_sequence_bundle(
        table, a.output, a.manifest, sequence_len=a.sequence_len, intervention_log=intervention
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
