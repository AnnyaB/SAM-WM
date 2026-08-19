from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SupportConformalCalibrator:
    bin_edges: np.ndarray
    quantiles: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls,
        residuals: np.ndarray,
        support_scores: np.ndarray,
        *,
        alpha: float = 0.1,
        bins: int = 5,
    ) -> SupportConformalCalibrator:
        r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
        s = np.asarray(support_scores, dtype=np.float64).reshape(-1)
        if r.size != s.size or r.size == 0:
            raise ValueError("residuals and support_scores must be aligned and non-empty")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        edges = np.quantile(s, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        q = np.empty(bins, dtype=np.float64)
        global_q = float(np.quantile(r, 1.0 - alpha, method="higher"))
        for i in range(bins):
            m = (s > edges[i]) & (s <= edges[i + 1])
            q[i] = float(np.quantile(r[m], 1.0 - alpha, method="higher")) if np.any(m) else global_q
        return cls(edges, q, alpha)

    def radius(self, support_score: float) -> float:
        idx = int(np.searchsorted(self.bin_edges[1:-1], support_score, side="right"))
        return float(self.quantiles[idx])
