from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    mae: float
    rmse: float
    interval_90_coverage: float
    interval_90_width: float


def forecast_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray,
) -> ForecastMetrics:
    y = np.asarray(y_true, dtype=float)[mask]
    mu = np.asarray(mean, dtype=float)[mask]
    sigma = np.maximum(np.asarray(std, dtype=float)[mask], 1e-8)
    if y.size == 0:
        raise ValueError("no valid evaluation points")
    error = y - mu
    z = 1.6448536269514722
    lo, hi = mu - z * sigma, mu + z * sigma
    return ForecastMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        interval_90_coverage=float(np.mean((y >= lo) & (y <= hi))),
        interval_90_width=float(np.mean(hi - lo)),
    )
