from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from coolworld.benchmarks import load_fairurbtemp, load_freiburg, load_novisad, save_manifest
from coolworld.config import load_yaml
from coolworld.experiment import calibration_from_split, evaluate_split


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a frozen SAM-WM checkpoint on ID or zero-shot OOD data."
    )
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", choices=["freiburg", "novisad", "fairurbtemp"], required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--city", default=None)
    p.add_argument("--out", default="artifacts/eval")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    if a.data == "freiburg":
        ds = load_freiburg(a.root or cfg["data_root"], k=int(cfg["graph_k"]))
        radius = calibration_from_split(ds, cfg, a.checkpoint)
        split = tuple(cfg["splits"]["test"])
        protocol = "ID_FINAL_TEST_AFTER_VALIDATION_CALIBRATION"
    elif a.data == "novisad":
        ds = load_novisad(a.root or "data/novisad", k=int(cfg["graph_k"]))
        # Zero-shot: Freiburg validation radius is not recalibrated on OOD target labels.
        src = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
        radius = calibration_from_split(src, cfg, a.checkpoint)
        split = (str(ds.timestamps[48]), str(ds.timestamps[-1]))
        protocol = "ZERO_SHOT_NO_FINETUNE_NO_OOD_RECALIBRATION"
    else:
        if not a.root:
            raise SystemExit("--root is required for extracted FAIRUrbTemp DOI 10.48620/93247")
        ds = load_fairurbtemp(a.root, city=a.city, k=int(cfg["graph_k"]))
        src = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
        radius = calibration_from_split(src, cfg, a.checkpoint)
        split = (str(ds.timestamps[48]), str(ds.timestamps[-1]))
        protocol = "ZERO_SHOT_NO_FINETUNE_NO_OOD_RECALIBRATION"
    metrics = evaluate_split(ds, cfg, a.checkpoint, split, radius=radius)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_manifest(ds, out / f"{a.data}_manifest.json")
    payload = {
        "dataset": a.data,
        "protocol": protocol,
        "checkpoint": str(a.checkpoint),
        "metrics": asdict(metrics),
    }
    (out / f"{a.data}_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
