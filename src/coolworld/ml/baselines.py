from __future__ import annotations

import numpy as np


def persistence_forecast(
    context_dynamic: np.ndarray,
    *,
    temperature_index: int,
    pred_len: int,
) -> np.ndarray:
    """Repeat the last observed temperature for every future horizon.

    Input shape: [B, T, N, F]. Output shape: [B, H, N].
    This is a deterministic baseline over real input data; it creates no
    synthetic empirical evidence.
    """
    x = np.asarray(context_dynamic, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError("context_dynamic must have shape [B,T,N,F]")
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")
    if not 0 <= temperature_index < x.shape[-1]:
        raise ValueError("temperature_index out of range")
    last = x[:, -1, :, temperature_index]
    return np.repeat(last[:, None, :], pred_len, axis=1)


def linear_trend_forecast(
    context_dynamic: np.ndarray,
    *,
    temperature_index: int,
    pred_len: int,
) -> np.ndarray:
    """Per-tile least-squares linear trend extrapolation baseline.

    It is intentionally simple and cheap, making it a useful sanity baseline
    against a learned world model. It must not be interpreted as a physical
    intervention model.
    """
    x = np.asarray(context_dynamic, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError("context_dynamic must have shape [B,T,N,F]")
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")
    if not 0 <= temperature_index < x.shape[-1]:
        raise ValueError("temperature_index out of range")

    y = x[..., temperature_index]
    time = np.arange(y.shape[1], dtype=np.float32)
    centered = time - time.mean()
    denom = float(np.sum(centered**2))
    if denom <= 0:
        return persistence_forecast(
            x,
            temperature_index=temperature_index,
            pred_len=pred_len,
        )

    y_mean = y.mean(axis=1, keepdims=True)
    slope = np.sum((y - y_mean) * centered[None, :, None], axis=1) / denom
    intercept = y_mean[:, 0, :] - slope * time.mean()
    future_t = np.arange(y.shape[1], y.shape[1] + pred_len, dtype=np.float32)
    return intercept[:, None, :] + slope[:, None, :] * future_t[None, :, None]
