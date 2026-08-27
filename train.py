from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from coolworld.benchmarks import load_freiburg, save_manifest
from coolworld.config import load_yaml
from coolworld.experiment import train_model


def configure_reproducibility() -> None:
    """Request deterministic kernels where PyTorch provides them.

    Exact floating-point identity across different GPU models/software stacks is
    not promised; seeds, config, data hashes and held-out receipts remain the
    reproducibility record.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SAM-WM on the preregistered Freiburg development protocol."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/freiburg")
    args = parser.parse_args()
    configure_reproducibility()
    cfg = load_yaml(args.config)
    ds = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
    out = Path(args.out) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(ds, out / "dataset_manifest.json")
    ckpt = train_model(ds, cfg, out, args.seed)
    print(json.dumps({"checkpoint": str(ckpt), "dataset": ds.name, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
