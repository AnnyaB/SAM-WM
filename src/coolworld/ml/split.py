from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PartitionRange:
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def chronological_partition_ranges(
    samples: int,
    *,
    sequence_len: int,
    calibration_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[PartitionRange, PartitionRange, PartitionRange]:
    if samples < 3:
        raise ValueError("at least three sequence samples are required")
    if sequence_len < 2:
        raise ValueError("sequence_len must be at least 2")
    if not 0 < calibration_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("calibration/test fractions must lie in (0, 0.5)")
    if calibration_fraction + test_fraction >= 0.7:
        raise ValueError("calibration + test fraction is too large")

    purge = sequence_len - 1
    test_n = max(1, int(round(samples * test_fraction)))
    cal_n = max(1, int(round(samples * calibration_fraction)))

    test_start = samples - test_n
    cal_end = test_start - purge
    cal_start = cal_end - cal_n
    development_end = cal_start - purge

    if development_end < 2 or cal_start < 0 or cal_end <= cal_start:
        raise ValueError(
            "dataset too small for purged development/calibration/test split; "
            "collect more timestamps or reduce sequence length"
        )

    return (
        PartitionRange("development", 0, development_end),
        PartitionRange("calibration", cal_start, cal_end),
        PartitionRange("test", test_start, samples),
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sequence_bundle(
    dataset_npz: str | Path,
    dataset_manifest: str | Path,
    output_dir: str | Path,
    *,
    calibration_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    dataset_path = Path(dataset_npz)
    manifest_path = Path(dataset_manifest)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_hash = _sha256_file(dataset_path)
    if parent_hash != raw_manifest.get("file_sha256"):
        raise ValueError("parent dataset hash does not match manifest")

    bundle = np.load(dataset_path, allow_pickle=False)
    required = {"dynamic", "static", "actions", "mask"}
    missing = required.difference(bundle.files)
    if missing:
        raise ValueError(f"dataset NPZ missing arrays: {sorted(missing)}")

    arrays = {name: np.asarray(bundle[name]) for name in required}
    samples = int(arrays["dynamic"].shape[0])
    if any(int(value.shape[0]) != samples for value in arrays.values()):
        raise ValueError("all dataset arrays must share the sample dimension")

    sequence_len = int(raw_manifest.get("sequence_len", arrays["dynamic"].shape[1]))
    ranges = chronological_partition_ranges(
        samples,
        sequence_len=sequence_len,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
    )
    purge = sequence_len - 1

    outputs: dict[str, Any] = {
        "parent_dataset_id": raw_manifest.get("dataset_id"),
        "parent_file_sha256": parent_hash,
        "split_policy": "chronological_purged_development_calibration_test",
        "purge_windows": purge,
        "partitions": {},
    }

    for part in ranges:
        out_npz = output_root / f"{part.name}.npz"
        np.savez_compressed(
            out_npz,
            dynamic=arrays["dynamic"][part.start : part.end],
            static=arrays["static"][part.start : part.end],
            actions=arrays["actions"][part.start : part.end],
            mask=arrays["mask"][part.start : part.end],
        )
        out_hash = _sha256_file(out_npz)
        part_manifest = dict(raw_manifest)
        part_manifest.update(
            {
                "dataset_id": f"{raw_manifest['dataset_id']}-{part.name}-{out_hash[:12]}",
                "file_sha256": out_hash,
                "parent_dataset_id": raw_manifest["dataset_id"],
                "parent_file_sha256": parent_hash,
                "partition": part.name,
                "sample_range": [part.start, part.end - 1],
                "samples": part.size,
                "split_policy": outputs["split_policy"],
                "purge_windows": purge,
            }
        )
        out_manifest = output_root / f"{part.name}.manifest.json"
        out_manifest.write_text(
            json.dumps(part_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        outputs["partitions"][part.name] = {
            "npz": str(out_npz),
            "manifest": str(out_manifest),
            "file_sha256": out_hash,
            "samples": part.size,
            "sample_range": [part.start, part.end - 1],
        }

    split_manifest = output_root / "split_manifest.json"
    split_manifest.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    return outputs
