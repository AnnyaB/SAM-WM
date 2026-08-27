from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import sha256_file
from .research import RESEARCH_SEEDS


class PromotionError(RuntimeError):
    """Raised when a frozen research artifact cannot be promoted safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"artifact is not a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_deployment_seed(
    research_root: Path,
    source_sha_path: Path,
    out: Path,
) -> dict[str, Any]:
    """Choose the deployment seed using Freiburg validation only, before held-out access."""
    manifest = research_root / "PRE_FREEZE_MANIFEST.json"
    manifest_payload = _read_json(manifest)
    if manifest_payload.get("heldout_or_ood_accessed") is not False:
        raise PromotionError("pre-freeze manifest does not certify held-out isolation")

    rows: list[tuple[float, int, Path, str]] = []
    for seed in RESEARCH_SEEDS:
        metrics_path = research_root / "full" / f"seed_{seed}" / "validation_metrics.json"
        payload = _read_json(metrics_path)
        if payload.get("heldout_or_ood_accessed") is not False:
            raise PromotionError(f"seed {seed} validation artifact is not development-only")
        mae = float(payload.get("validation", {}).get("mae", float("nan")))
        checkpoint = research_root / "full" / f"seed_{seed}" / "best.pt"
        if not np.isfinite(mae) or not checkpoint.is_file():
            raise PromotionError(f"seed {seed} validation/checkpoint evidence incomplete")
        checkpoint_sha = sha256_file(checkpoint)
        if payload.get("checkpoint_sha256") != checkpoint_sha:
            raise PromotionError(f"seed {seed} checkpoint hash mismatch")
        rows.append((mae, seed, checkpoint, checkpoint_sha))

    mae, seed, checkpoint, checkpoint_sha = min(rows, key=lambda row: (row[0], row[1]))
    source_sha = source_sha_path.read_text(encoding="utf-8").strip()
    if len(source_sha) != 40:
        raise PromotionError("invalid frozen source SHA")

    selection = {
        "protocol": "SAM_WM_DEPLOYMENT_SELECTION_V1",
        "selection_rule": "minimum Freiburg validation MAE; tie-break by ascending frozen seed",
        "heldout_or_ood_used_for_selection": False,
        "source_sha": source_sha,
        "pre_freeze_manifest_sha256": sha256_file(manifest),
        "selected_seed": seed,
        "selected_validation_mae_c": mae,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "candidate_validation_mae_c": {str(row[1]): row[0] for row in rows},
    }
    _write_json(out, selection)
    return selection


def finalize_deployment_bundle(
    selection_path: Path,
    freeze_path: Path,
    eval_root: Path,
    deployment_root: Path,
) -> dict[str, Any]:
    """Promote one preselected checkpoint only after all frozen evaluations exist."""
    selection = _read_json(selection_path)
    if selection.get("protocol") != "SAM_WM_DEPLOYMENT_SELECTION_V1":
        raise PromotionError("invalid deployment selection protocol")
    if selection.get("heldout_or_ood_used_for_selection") is not False:
        raise PromotionError("deployment seed was not selected from validation only")

    freeze = _read_json(freeze_path)
    if freeze.get("protocol") != "SAM_WM_FINAL_FREEZE_V1":
        raise PromotionError("invalid final freeze protocol")
    seed = int(selection["selected_seed"])
    checkpoint = Path(str(selection["checkpoint"]))
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != selection.get("checkpoint_sha256"):
        raise PromotionError("selected checkpoint changed after preselection")
    frozen_sha = freeze.get("full_checkpoints", {}).get(f"seed_{seed}")
    if frozen_sha != checkpoint_sha:
        raise PromotionError("selected checkpoint is not the frozen checkpoint")

    names = ("freiburg_heldout", "novisad_heldout", "fairurbtemp_heldout")
    artifacts: dict[str, Path] = {
        name: eval_root / f"seed_{seed}" / f"{name}_metrics.json" for name in names
    }
    radii: list[float] = []
    hashes: dict[str, str] = {}
    for name, path in artifacts.items():
        payload = _read_json(path)
        if payload.get("checkpoint_sha256") != checkpoint_sha:
            raise PromotionError(f"{name} was not evaluated with selected frozen checkpoint")
        radius = float(payload.get("conformal_radius_c", float("nan")))
        if not np.isfinite(radius) or radius <= 0:
            raise PromotionError(f"{name} lacks a valid Freiburg-validation conformal radius")
        radii.append(radius)
        hashes[name] = sha256_file(path)
    if not np.allclose(radii, radii[0], rtol=0.0, atol=1e-9):
        raise PromotionError("evaluation artifacts disagree on frozen conformal radius")

    deployment_root.mkdir(parents=True, exist_ok=True)
    promoted_checkpoint = deployment_root / "best.pt"
    shutil.copyfile(checkpoint, promoted_checkpoint)
    if sha256_file(promoted_checkpoint) != checkpoint_sha:
        raise PromotionError("promoted checkpoint copy failed integrity check")

    calibration = {
        "protocol": "SAM_WM_DEPLOYMENT_CALIBRATION_V1",
        "checkpoint_sha256": checkpoint_sha,
        "conformal_radius_c": radii[0],
        "source": "Freiburg validation split-conformal residuals; frozen before OOD use",
        "selection_sha256": sha256_file(selection_path),
    }
    evaluation = {
        "protocol": "SAM_WM_DEPLOYMENT_EVIDENCE_V1",
        "checkpoint_sha256": checkpoint_sha,
        "source_sha": selection["source_sha"],
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "selection_sha256": sha256_file(selection_path),
        "required_evaluations": hashes,
        "claim_boundary": (
            "These artifacts establish the frozen benchmark evidence only; they do not establish "
            "a causal intervention effect or universal deployment safety."
        ),
    }
    _write_json(deployment_root / "calibration.json", calibration)
    _write_json(deployment_root / "evaluation.json", evaluation)

    result = {
        "protocol": "SAM_WM_DEPLOYMENT_PROMOTION_V1",
        "selected_seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "calibration_sha256": sha256_file(deployment_root / "calibration.json"),
        "evaluation_sha256": sha256_file(deployment_root / "evaluation.json"),
        "bundle_fingerprint": _sha256_text(
            checkpoint_sha
            + sha256_file(deployment_root / "calibration.json")
            + sha256_file(deployment_root / "evaluation.json")
        ),
    }
    _write_json(deployment_root / "PROMOTION_MANIFEST.json", result)
    return result
