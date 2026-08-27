from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coolworld import research
from coolworld.benchmarks import UrbanDataset
from coolworld.freiburg import EXPECTED_HOURS, EXPECTED_STATIONS


def test_research_contract_is_one_model_and_five_frozen_seeds():
    assert research.RESEARCH_SEEDS == (17, 29, 42, 73, 101)
    assert not hasattr(research, "ABLATIONS")
    assert not hasattr(research, "CONTROLS")


def test_freiburg_preflight_writes_evidence_without_heldout_access(tmp_path: Path):
    temperature = np.ones((EXPECTED_HOURS, EXPECTED_STATIONS), dtype=np.float32)
    ds = UrbanDataset(
        name="freiburg",
        timestamps=np.arange(EXPECTED_HOURS),
        temperature=temperature,
        rh=np.ones_like(temperature),
        observed_mask=np.ones_like(temperature, dtype=bool),
        station_ids=tuple(f"FR{i:04d}" for i in range(EXPECTED_STATIONS)),
        lat=np.linspace(47.9, 48.1, EXPECTED_STATIONS, dtype=np.float32),
        lon=np.linspace(7.7, 7.9, EXPECTED_STATIONS, dtype=np.float32),
        elevation=np.zeros(EXPECTED_STATIONS, dtype=np.float32),
        edge_index=torch.zeros((2, 4), dtype=torch.long),
        edge_attr=torch.zeros((4, 3), dtype=torch.float32),
        source="doi:10.5281/zenodo.12732565",
    )

    payload = research.preflight_dataset(ds, tmp_path)
    assert payload["protocol"] == "SAM_WM_FREIBURG_PREFLIGHT_V2"
    assert payload["dataset"] == "freiburg"
    assert payload["n_nodes"] == EXPECTED_STATIONS
    assert payload["n_timestamps"] == EXPECTED_HOURS
    assert payload["heldout_or_ood_accessed"] is False
    assert (tmp_path / "FREIBURG_PREFLIGHT.json").is_file()


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
