from __future__ import annotations

import json

import numpy as np
import pandas as pd

from coolworld.action_evidence import build_action_evidence


def study(path, effect: float) -> None:
    rng = np.random.default_rng(7)
    rows = []
    for horizon in range(1, 4):
        for _ in range(24):
            treated_pre = 35.0 + rng.normal(0, 0.1)
            control_pre = 35.0 + rng.normal(0, 0.1)
            rows.append(
                {
                    "horizon_hour": horizon,
                    "treated_pre_c": treated_pre,
                    "treated_post_c": treated_pre + effect + rng.normal(0, 0.05),
                    "control_pre_c": control_pre,
                    "control_post_c": control_pre + rng.normal(0, 0.05),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_action_artifact_requires_source_and_transfer_cooling(tmp_path):
    source = tmp_path / "source.csv"
    transfer = tmp_path / "transfer.csv"
    study(source, -2.0)
    study(transfer, -1.2)
    out = tmp_path / "actions.json"

    action = build_action_evidence(
        kind="shade",
        source_csv=source,
        transfer_csv=transfer,
        source_provenance="doi:source",
        transfer_provenance="doi:transfer",
        reference_coverage_fraction=0.4,
        coverage_tolerance=0.1,
        horizon=3,
        out=out,
        block=8,
        samples=200,
        seed=11,
        min_pairs=20,
        support_reference_pairs=20,
    )
    assert action["transfer_validated"] is True
    assert all(value < 0 for value in action["interval_high_c_by_horizon"])
    assert action["support_score"] == 1.0
    payload = json.loads(out.read_text())
    assert payload["protocol"] == "SAM_WM_CANDRA_ACTIONS_V1"
    assert payload["actions"]["shade"]["source_study"]["sha256"]
    assert payload["actions"]["shade"]["transfer_study"]["sha256"]
