import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from coolworld.ml.data import UrbanThermalSequenceDataset


def test_manifest_rejects_unapproved_evidence_kind(tmp_path: Path):
    p = tmp_path / "x.npz"
    np.savez_compressed(
        p,
        dynamic=np.zeros((1, 2, 1, 1), np.float32),
        static=np.zeros((1, 1, 0), np.float32),
        actions=np.zeros((1, 2, 1, 1), np.float32),
        mask=np.ones((1, 2, 1), bool),
    )
    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "dataset_id": "x",
                "file_sha256": sha256(p.read_bytes()).hexdigest(),
                "source_records": [{"kind": "synthetic", "content_sha256": "a" * 64}],
                "schema": {
                    "dynamic_features": ["temperature_c"],
                    "static_features": [],
                    "action_features": ["shade"],
                    "temperature_feature": "temperature_c",
                },
            }
        )
    )
    with pytest.raises(ValueError):
        UrbanThermalSequenceDataset(p, m, context_len=1, pred_len=1)
