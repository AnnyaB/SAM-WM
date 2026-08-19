import numpy as np
import pytest

from coolworld.interventions import Intervention, InterventionKind, encode_actions


def test_intervention_encoding_is_explicit_and_bounded():
    x = Intervention(InterventionKind.SHADE, 0.4, ("a", "b"))
    actions = encode_actions(["a", "b", "c"], [x])
    assert actions.shape == (3, 3)
    assert np.allclose(actions[:, 0], [0.4, 0.4, 0.0])


def test_unknown_tile_is_rejected():
    x = Intervention(InterventionKind.SHADE, 0.5, ("missing",))
    with pytest.raises(ValueError):
        encode_actions(["a"], [x])
