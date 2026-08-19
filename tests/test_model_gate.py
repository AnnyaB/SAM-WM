import json
from hashlib import sha256
from pathlib import Path

from coolworld.model_gate import validate_counterfactual_artifact


def test_model_gate_fails_closed_when_manifest_missing(tmp_path: Path) -> None:
    state = validate_counterfactual_artifact(tmp_path)
    assert not state.ready and state.status == "MODEL_NOT_READY"


def test_model_gate_requires_evidence_manifest(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.bin"
    ckpt.write_bytes(b"deterministic-unit-test-checkpoint")
    digest = sha256(ckpt.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "unit-test-only",
                "checkpoint": "model.bin",
                "checkpoint_sha256": digest,
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    state = validate_counterfactual_artifact(tmp_path)
    assert not state.ready and state.status == "MODEL_HAS_NO_EVIDENCE_MANIFEST"
