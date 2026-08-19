from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coolworld.ml.split import chronological_partition_ranges, split_sequence_bundle


def test_partition_ranges_are_purged() -> None:
    dev, cal, test = chronological_partition_ranges(100, sequence_len=18)
    assert dev.end + 17 == cal.start
    assert cal.end + 17 == test.start
    assert dev.size > 0 and cal.size > 0 and test.size > 0


def test_split_bundle_writes_hash_valid_partitions(tmp_path: Path) -> None:
    samples, steps, tiles = 100, 18, 2
    npz = tmp_path / "all.npz"
    np.savez_compressed(
        npz,
        dynamic=np.ones((samples, steps, tiles, 3), dtype=np.float32),
        static=np.ones((samples, tiles, 2), dtype=np.float32),
        actions=np.zeros((samples, steps, tiles, 3), dtype=np.float32),
        mask=np.ones((samples, steps, tiles), dtype=bool),
    )
    from coolworld.ml.data import sha256_file

    manifest = tmp_path / "all.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "real-unit-test",
                "file_sha256": sha256_file(npz),
                "source_records": [
                    {
                        "kind": "fortyguard_recorded_live",
                        "content_sha256": "a" * 64,
                        "source_reference": "unit-test-real-contract",
                    }
                ],
                "schema": {
                    "dynamic_features": ["temperature_c", "time_sin", "time_cos"],
                    "static_features": ["lat_scaled", "lon_scaled"],
                    "action_features": ["shade", "tree_canopy", "reflective_pavement"],
                    "temperature_feature": "temperature_c",
                },
                "sequence_len": steps,
            }
        ),
        encoding="utf-8",
    )
    out = split_sequence_bundle(npz, manifest, tmp_path / "splits")
    assert set(out["partitions"]) == {"development", "calibration", "test"}
    for name in out["partitions"]:
        assert (tmp_path / "splits" / f"{name}.npz").exists()
        assert (tmp_path / "splits" / f"{name}.manifest.json").exists()
