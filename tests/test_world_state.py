import pytest

from coolworld.world_state import WorldMode, WorldStatus, WorldViewState


def test_observed_mode_cannot_be_marked_modelled() -> None:
    with pytest.raises(ValueError):
        WorldViewState(WorldMode.OBSERVED, WorldStatus.READY, "observed", False)


def test_counterfactual_cannot_be_mislabelled_observed() -> None:
    with pytest.raises(ValueError):
        WorldViewState(WorldMode.COUNTERFACTUAL, WorldStatus.READY, "future", True)
