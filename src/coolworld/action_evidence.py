from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candra import difference_in_differences_block_bootstrap
from .evidence import sha256_file

REQUIRED_COLUMNS = {
    "horizon_hour",
    "treated_pre_c",
    "treated_post_c",
    "control_pre_c",
    "control_post_c",
}


class ActionEvidenceError(RuntimeError):
    """Raised when intervention evidence cannot support a deployable action artifact."""


def _study(
    path: Path,
    *,
    horizon: int,
    block: int,
    samples: int,
    seed: int,
    min_pairs: int,
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ActionEvidenceError(f"missing intervention columns: {sorted(missing)}")

    effects: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    statuses: list[str] = []
    counts: list[int] = []
    for hour in range(1, horizon + 1):
        rows = frame.loc[frame["horizon_hour"] == hour, sorted(REQUIRED_COLUMNS - {"horizon_hour"})]
        rows = rows.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rows) < min_pairs:
            raise ActionEvidenceError(
                f"horizon {hour} has {len(rows)} complete pairs; at least {min_pairs} required"
            )
        interval = difference_in_differences_block_bootstrap(
            rows["treated_pre_c"].to_numpy(),
            rows["treated_post_c"].to_numpy(),
            rows["control_pre_c"].to_numpy(),
            rows["control_post_c"].to_numpy(),
            block=min(block, len(rows)),
            samples=samples,
            seed=seed + hour,
        )
        effects.append(interval.effect_c)
        lows.append(interval.low_c)
        highs.append(interval.high_c)
        statuses.append(interval.status)
        counts.append(int(len(rows)))

    return {
        "effect_c_by_horizon": effects,
        "interval_low_c_by_horizon": lows,
        "interval_high_c_by_horizon": highs,
        "status_by_horizon": statuses,
        "pairs_by_horizon": counts,
        "sha256": sha256_file(path),
    }


def build_action_evidence(
    *,
    kind: str,
    source_csv: Path,
    transfer_csv: Path,
    source_provenance: str,
    transfer_provenance: str,
    reference_coverage_fraction: float,
    coverage_tolerance: float,
    horizon: int,
    out: Path,
    block: int = 24,
    samples: int = 2000,
    seed: int = 0,
    min_pairs: int = 48,
    support_reference_pairs: int = 168,
) -> dict[str, Any]:
    """Build one action only when source and independent transfer evidence both support cooling.

    `support_score` is an evidence-volume indicator, not a probability of truth. Causal validity
    still depends on the treated/control assumptions of both supplied studies.
    """
    kind = kind.strip()
    if not kind:
        raise ActionEvidenceError("action kind is required")
    if not 0 < reference_coverage_fraction <= 1:
        raise ActionEvidenceError("reference coverage must lie in (0,1]")
    if not 0 <= coverage_tolerance <= 1:
        raise ActionEvidenceError("coverage tolerance must lie in [0,1]")
    if horizon < 1 or min_pairs < 2 or support_reference_pairs < min_pairs:
        raise ActionEvidenceError("invalid horizon/evidence-size contract")

    source = _study(
        source_csv,
        horizon=horizon,
        block=block,
        samples=samples,
        seed=seed,
        min_pairs=min_pairs,
    )
    transfer = _study(
        transfer_csv,
        horizon=horizon,
        block=block,
        samples=samples,
        seed=seed + 10_000,
        min_pairs=min_pairs,
    )
    source_supported = all(status == "SUPPORTED_COOLING" for status in source["status_by_horizon"])
    transfer_supported = all(
        status == "SUPPORTED_COOLING" for status in transfer["status_by_horizon"]
    )
    transfer_validated = source_supported and transfer_supported
    minimum_pairs = min(*source["pairs_by_horizon"], *transfer["pairs_by_horizon"])
    support_score = min(1.0, minimum_pairs / float(support_reference_pairs))

    action = {
        "transfer_validated": transfer_validated,
        "effect_c_by_horizon": transfer["effect_c_by_horizon"],
        "interval_low_c_by_horizon": transfer["interval_low_c_by_horizon"],
        "interval_high_c_by_horizon": transfer["interval_high_c_by_horizon"],
        "support_score": support_score,
        "support_score_semantics": "min paired evidence volume / support_reference_pairs; not probability",
        "reference_coverage_fraction": reference_coverage_fraction,
        "coverage_tolerance": coverage_tolerance,
        "source": source_provenance,
        "transfer_source": transfer_provenance,
        "source_study": source,
        "transfer_study": transfer,
        "minimum_pairs_required_per_horizon": min_pairs,
        "support_reference_pairs": support_reference_pairs,
    }

    if out.exists():
        try:
            payload = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ActionEvidenceError("existing action artifact is invalid JSON") from exc
        if payload.get("protocol") != "SAM_WM_CANDRA_ACTIONS_V1":
            raise ActionEvidenceError("existing action artifact has incompatible protocol")
    else:
        payload = {"protocol": "SAM_WM_CANDRA_ACTIONS_V1", "actions": {}}
    actions = payload.setdefault("actions", {})
    if not isinstance(actions, dict):
        raise ActionEvidenceError("actions field is invalid")
    actions[kind] = action

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    return action
