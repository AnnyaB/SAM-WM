from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coolworld.sam import support_deficiency


@dataclass(frozen=True, slots=True)
class LocalActionSupport:
    covariance: np.ndarray
    deficiency: np.ndarray
    support_score: float
    neighbors: int


def local_action_support(
    query_context: np.ndarray,
    contexts: np.ndarray,
    actions: np.ndarray,
    *,
    candidate_action: np.ndarray | None = None,
    k: int = 64,
    tau: float = 0.05,
) -> LocalActionSupport:
    """Estimate empirical action support near a context using real logged actions."""

    q = np.asarray(query_context, dtype=np.float64)
    x = np.asarray(contexts, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    if x.ndim != 2 or a.ndim != 2 or x.shape[0] != a.shape[0]:
        raise ValueError("contexts/actions must be aligned 2D arrays")
    if q.shape != (x.shape[1],):
        raise ValueError("query context dimension mismatch")
    if x.shape[0] < 2:
        raise ValueError("at least two logged contexts are required")
    k_eff = min(max(2, k), x.shape[0])
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    dist = np.sum(((x - q) / scale) ** 2, axis=1)
    idx = np.argpartition(dist, k_eff - 1)[:k_eff]
    local = a[idx]
    covariance = np.cov(local, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    deficiency = support_deficiency(covariance, tau)
    # 1 = rich variation, 0 = no variation, averaged over action directions.
    variation_score = float(np.clip(1.0 - np.trace(deficiency) / deficiency.shape[0], 0.0, 1.0))
    support_score = variation_score
    if candidate_action is not None:
        candidate = np.asarray(candidate_action, dtype=np.float64).reshape(-1)
        if candidate.shape != (a.shape[1],):
            raise ValueError("candidate action dimension mismatch")
        mean = local.mean(axis=0)
        reg = covariance + tau * np.eye(covariance.shape[0])
        delta = candidate - mean
        distance2 = float(delta @ np.linalg.solve(reg, delta))
        proximity = float(np.exp(-0.5 * distance2 / max(1, a.shape[1])))
        support_score *= proximity
    return LocalActionSupport(
        covariance, deficiency, float(np.clip(support_score, 0.0, 1.0)), k_eff
    )
