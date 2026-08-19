from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    candidate_id: str
    predicted_delta_c: float
    interval_low_c: float
    interval_high_c: float
    support_score: float
    cost: float | None
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class CoolingDecision:
    selected_id: str | None
    status: str
    reason: str


def choose_cooling_action(
    outcomes: Iterable[CandidateOutcome],
    *,
    budget: float | None = None,
    min_support: float = 0.15,
) -> CoolingDecision:
    """Select the strongest evidence-supported cooling option.

    Cooling is represented as negative temperature change. The conservative
    comparison uses each candidate's *upper* interval bound: an action is only
    treated as cooling when that bound remains below zero.
    """

    eligible: list[CandidateOutcome] = []
    for c in outcomes:
        if c.conflict or c.support_score < min_support:
            continue
        if budget is not None and (c.cost is None or c.cost > budget):
            continue
        if c.interval_high_c < 0.0:
            eligible.append(c)
    if not eligible:
        return CoolingDecision(
            None,
            "NO_SUPPORTED_COOLING_ACTION",
            "No candidate passes support, uncertainty, conflict, and budget checks.",
        )
    # More negative upper bound = stronger cooling even under uncertainty.
    selected = min(
        eligible, key=lambda c: (c.interval_high_c, c.cost if c.cost is not None else float("inf"))
    )
    return CoolingDecision(
        selected.candidate_id,
        "SELECTED",
        "Selected by uncertainty-bounded cooling effect; cost breaks ties when available.",
    )
