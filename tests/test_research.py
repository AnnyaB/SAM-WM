from __future__ import annotations

import json
from pathlib import Path

import pytest

from coolworld import research


def test_research_contract_is_one_model_and_five_frozen_seeds():
    assert research.RESEARCH_SEEDS == (17, 29, 42, 73, 101)
    assert not hasattr(research, "ABLATIONS")
    assert not hasattr(research, "CONTROLS")


def test_pre_freeze_manifest_contains_only_samwm_validation_evidence(tmp_path: Path):
    root = tmp_path / "research"
    for seed in research.RESEARCH_SEEDS:
        path = root / f"seed_{seed}" / "validation_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": "SAM-WM",
                    "seed": seed,
                    "heldout_or_ood_accessed": False,
                }
            ),
            encoding="utf-8",
        )
    config = tmp_path / "train.yaml"
    config.write_text("model: SAM-WM\n", encoding="utf-8")

    research.write_pre_freeze_manifest(
        root,
        config_path=config,
        dataset_name="freiburg",
    )
    manifest = root / "PRE_FREEZE_MANIFEST.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["protocol"] == "SAM_WM_PRE_FREEZE_V2"
    assert payload["model"] == "SAM-WM"
    assert payload["heldout_or_ood_accessed"] is False
    assert len(payload["validation_artifacts"]) == len(research.RESEARCH_SEEDS)
    assert all("seed_" in key for key in payload["validation_artifacts"])


def test_pre_freeze_manifest_refuses_missing_seed(tmp_path: Path):
    root = tmp_path / "research"
    config = tmp_path / "train.yaml"
    config.write_text("model: SAM-WM\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        research.write_pre_freeze_manifest(
            root,
            config_path=config,
            dataset_name="freiburg",
        )
