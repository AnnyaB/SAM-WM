from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DifferenceInDifferencesResult:
    effect: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    bootstrap_samples: int


def difference_in_differences(
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> DifferenceInDifferencesResult:
    """Matched before/after effect estimate with paired bootstrap uncertainty.

    Arrays contain independent matched units or repeated intervention sites. The
    caller is responsible for constructing controls without looking at post-
    treatment outcomes and for verifying design assumptions/pre-trends.
    """

    tp = np.asarray(treated_pre, dtype=float).reshape(-1)
    tq = np.asarray(treated_post, dtype=float).reshape(-1)
    cp = np.asarray(control_pre, dtype=float).reshape(-1)
    cq = np.asarray(control_post, dtype=float).reshape(-1)
    if tp.size != tq.size or cp.size != cq.size or tp.size == 0 or cp.size == 0:
        raise ValueError("treated/control pre/post arrays must be aligned and non-empty")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    effect = float(np.mean(tq - tp) - np.mean(cq - cp))
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=float)
    for i in range(bootstrap_samples):
        ti = rng.integers(0, tp.size, tp.size)
        ci = rng.integers(0, cp.size, cp.size)
        boot[i] = np.mean(tq[ti] - tp[ti]) - np.mean(cq[ci] - cp[ci])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return DifferenceInDifferencesResult(
        effect, float(lo), float(hi), tp.size, cp.size, bootstrap_samples
    )
