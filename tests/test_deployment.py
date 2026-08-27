from __future__ import annotations

import json
from pathlib import Path

import pytest

import coolworld.deployment as deployment
from coolworld.evidence import sha256_file


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deployment_bundle_is_bound_to_checkpoint_and_all_evaluations(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    calibration = tmp_path / "calibration.json"
    evaluation = tmp_path / "evaluation.json"
    write_json(
        calibration,
        {
            "protocol": "SAM_WM_DEPLOYMENT_CALIBRATION_V1",
            "checkpoint_sha256": checkpoint_sha,
            "conformal_radius_c": 1.1,
        },
    )
    write_json(
        evaluation,
        {
            "protocol": "SAM_WM_DEPLOYMENT_EVIDENCE_V1",
            "checkpoint_sha256": checkpoint_sha,
            "required_evaluations": {
                "freiburg_heldout": "a" * 64,
                "novisad_heldout": "b" * 64,
                "fairurbtemp_heldout": "c" * 64,
            },
        },
    )
    bundle = deployment.validate_deployment_bundle(checkpoint, calibration, evaluation)
    assert bundle.checkpoint_sha256 == checkpoint_sha
    assert bundle.conformal_radius_c == 1.1

    broken = json.loads(evaluation.read_text())
    broken["checkpoint_sha256"] = "0" * 64
    write_json(evaluation, broken)
    with pytest.raises(deployment.DeploymentError, match="EVALUATION_CHECKPOINT_HASH_MISMATCH"):
        deployment.validate_deployment_bundle(checkpoint, calibration, evaluation)


def test_canonical_map_data_rejects_duplicate_ids_and_missing_temperature():
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    duplicate = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "x", "properties": {"average_temperature": 30}, "geometry": geometry},
            {"type": "Feature", "id": "x", "properties": {"average_temperature": 31}, "geometry": geometry},
        ],
    }
    with pytest.raises(deployment.DeploymentError, match="DUPLICATE_TILE_ID"):
        deployment.canonical_map_data(duplicate)

    missing = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": "x", "properties": {}, "geometry": geometry}],
    }
    with pytest.raises(deployment.DeploymentError, match="OBSERVED_TEMPERATURE_MISSING"):
        deployment.canonical_map_data(missing)
