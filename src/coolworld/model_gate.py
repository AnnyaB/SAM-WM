from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelReadiness:
    ready: bool
    status: str
    model_id: str | None = None
    checkpoint_sha256: str | None = None


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_counterfactual_artifact(root: Path) -> ModelReadiness:
    """Fail closed unless checkpoint, hash, and real-evidence manifest agree."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ModelReadiness(False, "MODEL_NOT_READY")
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_id = str(manifest["model_id"])
        checkpoint_rel = str(manifest["checkpoint"])
        expected_hash = str(manifest["checkpoint_sha256"])
        evidence = manifest["evidence"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ModelReadiness(False, "INVALID_MODEL_MANIFEST")
    if not isinstance(evidence, list) or not evidence:
        return ModelReadiness(False, "MODEL_HAS_NO_EVIDENCE_MANIFEST")
    for item in evidence:
        if not isinstance(item, dict) or not item.get("content_sha256") or not item.get("kind"):
            return ModelReadiness(False, "INVALID_MODEL_EVIDENCE_MANIFEST")
    checkpoint_path = (root / checkpoint_rel).resolve()
    try:
        checkpoint_path.relative_to(root.resolve())
    except ValueError:
        return ModelReadiness(False, "INVALID_CHECKPOINT_PATH")
    if not checkpoint_path.is_file():
        return ModelReadiness(False, "CHECKPOINT_MISSING")
    actual_hash = _sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        return ModelReadiness(False, "CHECKPOINT_HASH_MISMATCH")
    return ModelReadiness(True, "READY", model_id=model_id, checkpoint_sha256=actual_hash)
