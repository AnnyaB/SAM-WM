from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .metrics import ForecastMetrics, forecast_metrics


def extreme_heat_mask(
    truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quantile: float,
) -> tuple[np.ndarray, float]:
    """Select the real high-temperature tail from observed evaluation targets."""
    y = np.asarray(truth, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if y.shape != mask.shape:
        raise ValueError("truth and valid_mask must have identical shapes")
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must lie in (0.5, 1.0)")
    valid = y[mask]
    if valid.size == 0:
        raise ValueError("no valid temperatures available")
    threshold = float(np.quantile(valid, quantile))
    return mask & (y >= threshold), threshold


def evaluate_temperature_tail(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quantile: float,
) -> dict[str, object]:
    tail, threshold = extreme_heat_mask(truth, valid_mask, quantile=quantile)
    metrics: ForecastMetrics = forecast_metrics(truth, mean, std, tail)
    return {
        "quantile": quantile,
        "threshold_c": threshold,
        "points": int(tail.sum()),
        **asdict(metrics),
    }
