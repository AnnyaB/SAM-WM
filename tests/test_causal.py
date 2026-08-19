import numpy as np

from coolworld.research.causal import difference_in_differences


def test_did_recovers_known_difference_from_fixed_arrays():
    result = difference_in_differences(
        np.array([40.0, 41.0, 42.0]),
        np.array([38.0, 39.0, 40.0]),
        np.array([40.0, 41.0, 42.0]),
        np.array([40.0, 41.0, 42.0]),
        bootstrap_samples=200,
        seed=7,
    )
    assert result.effect == -2.0
