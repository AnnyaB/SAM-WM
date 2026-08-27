from __future__ import annotations

import json
from pathlib import Path

from coolworld.evidence import sha256_file
from coolworld.promotion import finalize_deployment_bundle, select_deployment_seed
from coolworld.research import RESEARCH_SEEDS


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preselection_uses_validation_only_and_finalize_preserves_hashes(tmp_path: Path):
    research = tmp_path / "research"
    write_json(
        research / "PRE_FREEZE_MANIFEST.json",
        {
            "protocol": "SAM_WM_PRE_FREEZE_V2",
            "model": "SAM-WM",
            "heldout_or_ood_accessed": False,
        },
    )
    source = tmp_path / "FROZEN_SOURCE_SHA.txt"
    source.write_text("a" * 40 + "\n", encoding="utf-8")

    maes = {17: 2.0, 29: 1.8, 42: 1.2, 73: 1.5, 101: 1.9}
    checkpoint_hashes = {}
    for seed in RESEARCH_SEEDS:
        root = research / f"seed_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_hashes[f"seed_{seed}"] = sha256_file(checkpoint)
        write_json(
            root / "validation_metrics.json",
            {
                "model": "SAM-WM",
                "heldout_or_ood_accessed": False,
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation": {"mae": maes[seed]},
            },
        )

    selection_path = tmp_path / "DEPLOYMENT_SELECTION.json"
    selection = select_deployment_seed(research, source, selection_path)
    assert selection["model"] == "SAM-WM"
    assert selection["selected_seed"] == 42
    assert selection["heldout_or_ood_used_for_selection"] is False

    freeze = tmp_path / "FREEZE_MANIFEST.json"
    write_json(
        freeze,
        {"protocol": "SAM_WM_FINAL_FREEZE_V1", "full_checkpoints": checkpoint_hashes},
    )
    eval_root = tmp_path / "eval"
    for name in ("freiburg_heldout", "novisad_heldout", "fairurbtemp_heldout"):
        write_json(
            eval_root / "seed_42" / f"{name}_metrics.json",
            {
                "checkpoint_sha256": selection["checkpoint_sha256"],
                "conformal_radius_c": 1.25,
            },
        )

    deployment = tmp_path / "deployment"
    result = finalize_deployment_bundle(selection_path, freeze, eval_root, deployment)
    assert result["model"] == "SAM-WM"
    assert result["selected_seed"] == 42
    assert sha256_file(deployment / "best.pt") == selection["checkpoint_sha256"]
    assert json.loads((deployment / "calibration.json").read_text())["conformal_radius_c"] == 1.25
    evidence = json.loads((deployment / "evaluation.json").read_text())
    assert set(evidence["required_evaluations"]) == {
        "freiburg_heldout",
        "novisad_heldout",
        "fairurbtemp_heldout",
    }
