from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from .schema import FeatureSchema

_SUPPORTED_KNOWN_FUTURE = ("time_sin", "time_cos")


def known_future_indices(schema: FeatureSchema) -> tuple[int, ...]:
    indices = tuple(
        schema.dynamic_features.index(name)
        for name in _SUPPORTED_KNOWN_FUTURE
        if name in schema.dynamic_features
    )
    if not indices:
        raise ValueError("schema must contain at least one supported known-future feature")
    return indices


def future_known_from_dynamic(dynamic: np.ndarray, schema: FeatureSchema) -> np.ndarray:
    return np.asarray(dynamic[..., list(known_future_indices(schema))], dtype=np.float32)


def extrapolate_time_features(
    last_timestamp: datetime,
    *,
    steps: int,
    cadence_minutes: float,
    tiles: int,
    schema: FeatureSchema,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if steps <= 0 or cadence_minutes <= 0 or tiles <= 0:
        raise ValueError("steps, cadence_minutes and tiles must be positive")
    names = tuple(schema.dynamic_features[i] for i in known_future_indices(schema))
    rows: list[np.ndarray] = []
    timestamps: list[str] = []
    for step in range(1, steps + 1):
        ts = last_timestamp + timedelta(minutes=cadence_minutes * step)
        hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
        values = {
            "time_sin": math.sin(2 * math.pi * hour / 24.0),
            "time_cos": math.cos(2 * math.pi * hour / 24.0),
        }
        rows.append(
            np.tile(np.asarray([values[name] for name in names], dtype=np.float32), (tiles, 1))
        )
        timestamps.append(ts.isoformat())
    return np.stack(rows), tuple(timestamps)
