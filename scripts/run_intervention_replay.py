from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from coolworld.research.causal import difference_in_differences


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate a real matched intervention replay from a prepared Parquet file."
    )
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", default="outputs/intervention_replay.json")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    df = pd.read_parquet(a.parquet)
    required = {"group", "pre_temperature_c", "post_temperature_c"}
    if not required.issubset(df.columns):
        raise SystemExit(f"missing columns: {sorted(required.difference(df.columns))}")
    treated = df[df["group"] == "treated"]
    control = df[df["group"] == "control"]
    result = difference_in_differences(
        treated["pre_temperature_c"].to_numpy(),
        treated["post_temperature_c"].to_numpy(),
        control["pre_temperature_c"].to_numpy(),
        control["post_temperature_c"].to_numpy(),
        bootstrap_samples=a.bootstrap,
        seed=a.seed,
    )
    payload = result.__dict__
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
