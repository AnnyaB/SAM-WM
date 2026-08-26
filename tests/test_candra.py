import numpy as np

from coolworld.candra import conservative_action_status, difference_in_differences_block_bootstrap


def test_candra_abstains_without_support():
    assert conservative_action_status(-2, -1, 0.01) == "ABSTAIN_INSUFFICIENT_SUPPORT"


def test_did_detects_clear_cooling():
    n = 100
    tp = np.zeros(n)
    tq = np.full(n, -2.0)
    cp = np.zeros(n)
    cq = np.zeros(n)
    r = difference_in_differences_block_bootstrap(tp, tq, cp, cq, block=10, samples=300, seed=1)
    assert r.high_c < 0 and r.status == "SUPPORTED_COOLING"
