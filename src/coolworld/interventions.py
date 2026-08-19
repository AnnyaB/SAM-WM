from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class InterventionKind(StrEnum):
    SHADE = "shade"
    TREE_CANOPY = "tree_canopy"
    REFLECTIVE_PAVEMENT = "reflective_pavement"


ACTION_DIM = 3
_ACTION_INDEX = {
    InterventionKind.SHADE: 0,
    InterventionKind.TREE_CANOPY: 1,
    InterventionKind.REFLECTIVE_PAVEMENT: 2,
}


@dataclass(frozen=True, slots=True)
class Intervention:
    """A proposed or observed physical cooling intervention.

    `coverage_fraction` is the fraction of the target tile footprint to which the
    intervention is applied, represented in [0, 1]. It is a physical action
    descriptor, not a temperature-effect label. Real intervention records must
    derive the same quantity from mapped intervention geometry.
    """

    kind: InterventionKind
    coverage_fraction: float
    tile_ids: tuple[str, ...]
    cost: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage_fraction <= 1.0:
            raise ValueError("intervention coverage_fraction must be in [0, 1]")
        if not self.tile_ids:
            raise ValueError("intervention must target at least one tile")
        if self.cost is not None and self.cost < 0:
            raise ValueError("cost cannot be negative")

    def action_vector(self) -> np.ndarray:
        vec = np.zeros(ACTION_DIM, dtype=np.float32)
        vec[_ACTION_INDEX[self.kind]] = self.coverage_fraction
        return vec


def encode_actions(tile_ids: Iterable[str], interventions: Iterable[Intervention]) -> np.ndarray:
    ids = [str(x) for x in tile_ids]
    by_id = {tile_id: i for i, tile_id in enumerate(ids)}
    actions = np.zeros((len(ids), ACTION_DIM), dtype=np.float32)
    for intervention in interventions:
        vector = intervention.action_vector()
        for tile_id in intervention.tile_ids:
            if tile_id not in by_id:
                raise ValueError(f"intervention targets unknown tile_id={tile_id}")
            actions[by_id[tile_id]] += vector
    np.clip(actions, 0.0, 1.0, out=actions)
    return actions
