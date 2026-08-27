from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from coolworld.benchmarks import load_fairurbtemp, load_freiburg, load_novisad, save_manifest
from coolworld.config import load_yaml
from coolworld.experiment import calibration_from_split, evaluate_split


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_heldout(receipt: Path, payload: dict) -> None:
    """Atomically open a held-out evaluation exactly once for this output directory."""
    receipt.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(receipt, flags, 0o644)
    except FileExistsError as exc:
        raise SystemExit(
            f"held-out receipt already exists: {receipt}; refusing repeated label access"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen SAM-WM checkpoint without silently re-opening held-out labels."
    )
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", choices=["freiburg", "novisad", "fairurbtemp"], required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--city", default=None)
    parser.add_argument("--split", choices=["validation", "heldout"], default="heldout")
    parser.add_argument("--open-heldout", action="store_true")
    parser.add_argument("--out", default="artifacts/eval")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.data == "freiburg":
        ds = load_freiburg(args.root or cfg["data_root"], k=int(cfg["graph_k"]))
        if args.split == "validation":
            split = tuple(cfg["splits"]["val"])
            radius = None
            protocol = "FREIBURG_VALIDATION_MODEL_SELECTION_ONLY"
        else:
            if not args.open_heldout:
                raise SystemExit("Freiburg final test requires --open-heldout")
            split = tuple(cfg["splits"]["test"])
            radius = calibration_from_split(ds, cfg, args.checkpoint)
            protocol = "FROZEN_ID_FINAL_TEST_AFTER_VALIDATION_CALIBRATION"
    elif args.data == "novisad":
        if args.split != "heldout" or not args.open_heldout:
            raise SystemExit("Novi Sad is zero-shot held-out data; use --split heldout --open-heldout")
        ds = load_novisad(args.root or "data/novisad", k=int(cfg["graph_k"]))
        source = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
        radius = calibration_from_split(source, cfg, args.checkpoint)
        split = (str(ds.timestamps[48]), str(ds.timestamps[-1]))
        protocol = "ZERO_SHOT_NO_FINETUNE_NO_OOD_RECALIBRATION"
    else:
        if args.split != "heldout" or not args.open_heldout:
            raise SystemExit("FAIRUrbTemp is zero-shot held-out data; use --split heldout --open-heldout")
        if not args.root:
            raise SystemExit("--root is required for extracted FAIRUrbTemp DOI 10.48620/93247")
        ds = load_fairurbtemp(args.root, city=args.city, k=int(cfg["graph_k"]))
        source = load_freiburg(cfg["data_root"], k=int(cfg["graph_k"]))
        radius = calibration_from_split(source, cfg, args.checkpoint)
        split = (str(ds.timestamps[48]), str(ds.timestamps[-1]))
        protocol = "ZERO_SHOT_NO_FINETUNE_NO_OOD_RECALIBRATION_QC_FILTERED"

    if args.split == "heldout":
        receipt = out / f"{args.data}_HELDOUT_OPEN.json"
        claim_heldout(
            receipt,
            {
                "status": "OPENED_BEFORE_LABEL_ACCESS",
                "opened_at_utc": datetime.now(UTC).isoformat(),
                "dataset": args.data,
                "dataset_source": ds.source,
                "protocol": protocol,
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "config_sha256": sha256_file(args.config),
                "split": list(split),
            },
        )

    metrics = evaluate_split(ds, cfg, args.checkpoint, split, radius=radius)
    save_manifest(ds, out / f"{args.data}_manifest.json")
    payload = {
        "dataset": args.data,
        "dataset_source": ds.source,
        "protocol": protocol,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "metrics": asdict(metrics),
    }
    metrics_path = out / f"{args.data}_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
