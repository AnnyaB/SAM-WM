from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .benchmarks import UrbanDataset, load_freiburg, save_manifest
from .config import load_yaml
from .experiment import evaluate_split, train_model

RESEARCH_SEEDS = (17, 29, 42, 73, 101)


def configure_reproducibility() -> None:
    """Request deterministic kernels where PyTorch provides them."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_seed(
    ds: UrbanDataset,
    cfg: dict[str, Any],
    root: Path,
    seed: int,
) -> dict[str, Any]:
    """Train one full SAM-WM seed and score Freiburg validation only."""
    out = root / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(ds, out / "dataset_manifest.json")
    checkpoint = train_model(ds, cfg, out, seed)
    metrics = evaluate_split(
        ds,
        cfg,
        checkpoint,
        tuple(cfg["splits"]["val"]),
        radius=None,
    )
    payload = {
        "model": "SAM-WM",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "validation": asdict(metrics),
        "heldout_or_ood_accessed": False,
    }
    _write_json(out / "validation_metrics.json", payload)
    return payload


def write_pre_freeze_manifest(
    root: Path,
    *,
    config_path: Path,
    dataset_name: str,
) -> None:
    expected = [
        root / f"seed_{seed}" / "validation_metrics.json" for seed in RESEARCH_SEEDS
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "SAM-WM pre-freeze evidence is incomplete; refusing to seal manifest: "
            + ", ".join(missing)
        )
    manifest = {
        "protocol": "SAM_WM_PRE_FREEZE_V2",
        "model": "SAM-WM",
        "dataset": dataset_name,
        "seeds": list(RESEARCH_SEEDS),
        "config_sha256": _sha256_file(config_path),
        "research_py_sha256": _sha256_file(Path(__file__)),
        "validation_artifacts": {
            str(path.relative_to(root)): _sha256_file(path) for path in sorted(expected)
        },
        "heldout_or_ood_accessed": False,
        "rule": (
            "Only full SAM-WM is trained before freeze. Freiburg held-out, Novi Sad and "
            "FAIRUrbTemp targets remain forbidden until the reported run is frozen."
        ),
    }
    _write_json(root / "PRE_FREEZE_MANIFEST.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the frozen full SAM-WM seed suite on Freiburg development data."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--out", default="artifacts/research")
    args = parser.parse_args()

    configure_reproducibility()
    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    ds = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    for seed in RESEARCH_SEEDS:
        result = run_seed(ds, cfg, root, seed)
        print(
            json.dumps(
                {
                    "model": "SAM-WM",
                    "seed": seed,
                    "validation_mae": result["validation"]["mae"],
                }
            ),
            flush=True,
        )

    write_pre_freeze_manifest(root, config_path=config_path, dataset_name=ds.name)


if __name__ == "__main__":
    main()
