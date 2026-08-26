from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.benchmarks import load_freiburg, save_manifest
from coolworld.config import load_yaml
from coolworld.experiment import train_model


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train SAM-WM on the preregistered Freiburg development protocol."
    )
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="artifacts/freiburg")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    ds = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
    out = Path(a.out) / f"seed_{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(ds, out / "dataset_manifest.json")
    ckpt = train_model(ds, cfg, out, a.seed)
    print(json.dumps({"checkpoint": str(ckpt), "dataset": ds.name, "seed": a.seed}, indent=2))


if __name__ == "__main__":
    main()
