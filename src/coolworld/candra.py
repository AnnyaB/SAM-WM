from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectInterval:
    effect_c: float
    low_c: float
    high_c: float
    status: str


def difference_in_differences_block_bootstrap(
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray,
    *,
    block: int = 24,
    samples: int = 2000,
    seed: int = 0,
) -> EffectInterval:
    """Temporal block-bootstrap DiD reference estimator."""
    tp, tq, cp, cq = [
        np.asarray(x, float).reshape(-1)
        for x in (treated_pre, treated_post, control_pre, control_post)
    ]
    if min(map(len, (tp, tq, cp, cq))) == 0 or len(tp) != len(tq) or len(cp) != len(cq):
        raise ValueError("aligned non-empty treated/control pre/post arrays required")
    if samples < 200 or block < 1:
        raise ValueError("samples>=200 and block>=1 required")
    td, cd = tq - tp, cq - cp
    effect = float(td.mean() - cd.mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(samples):

        def resample(x: np.ndarray) -> np.ndarray:
            out = []
            while len(out) < len(x):
                s = int(rng.integers(0, max(1, len(x) - block + 1)))
                out.extend(x[s : s + block].tolist())
            return np.asarray(out[: len(x)])

        boot.append(resample(td).mean() - resample(cd).mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    status = "SUPPORTED_COOLING" if high < 0 else "NOT_ESTABLISHED"
    return EffectInterval(effect, float(low), float(high), status)


def conservative_action_status(
    low_c: float, high_c: float, support: float, min_support: float = 0.15
) -> str:
    if not np.isfinite([low_c, high_c, support]).all():
        return "ABSTAIN_INVALID_EVIDENCE"
    if support < min_support:
        return "ABSTAIN_INSUFFICIENT_SUPPORT"
    if high_c < 0:
        return "SUPPORTED_COOLING"
    if low_c > 0:
        return "SUPPORTED_WARMING_RISK"
    return "ABSTAIN_UNCERTAIN_EFFECT"
