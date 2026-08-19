from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorldMode(StrEnum):
    OBSERVED = "observed"
    VALIDATED_REPLAY = "validated_replay"
    COUNTERFACTUAL = "counterfactual"


class WorldStatus(StrEnum):
    READY = "READY"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    REPLAY_NOT_AVAILABLE = "REPLAY_NOT_AVAILABLE"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class WorldViewState:
    mode: WorldMode
    status: WorldStatus
    label: str
    is_observed: bool

    def __post_init__(self) -> None:
        if self.mode is WorldMode.OBSERVED and not self.is_observed:
            raise ValueError("observed mode must be marked observed")
        if self.mode is not WorldMode.OBSERVED and self.is_observed:
            raise ValueError("replay/counterfactual mode must not be labelled observed")
