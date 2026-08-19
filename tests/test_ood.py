import numpy as np

from coolworld.ml.ood import extreme_heat_mask


def test_extreme_heat_mask_selects_high_tail() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    valid = np.array([True, True, True, True])
    mask, threshold = extreme_heat_mask(truth, valid, quantile=0.75)
    assert threshold >= 3.0
    assert mask[-1]
