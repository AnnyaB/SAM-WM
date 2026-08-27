from __future__ import annotations

import numpy as np
import torch

from coolworld.benchmarks import UrbanDataset
from coolworld.experiment import derive_source_bound, fit_normalizer, normalized_dynamic


def dataset() -> UrbanDataset:
    hours = 360
    nodes = 4
    timestamps = np.datetime64("2023-01-01T00") + np.arange(hours).astype("timedelta64[h]")
    phase = np.arange(hours, dtype=np.float32)[:, None]
    temperature = 20.0 + 3.0 * np.sin(2 * np.pi * phase / 24.0)
    temperature = np.repeat(temperature, nodes, axis=1).astype(np.float32)
    rh = np.full_like(temperature, np.nan)
    observed = np.ones_like(temperature, dtype=bool)
    lat = np.array([48.0, 48.001, 48.002, 48.003], np.float32)
    lon = np.array([7.80, 7.801, 7.802, 7.803], np.float32)
    elevation = np.array([250.0, 255.0, 260.0, 265.0], np.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 0.0, 0.1]] * 3, dtype=torch.float32)
    return UrbanDataset(
        "unit_fixture",
        timestamps,
        temperature,
        rh,
        observed,
        tuple(f"s{i}" for i in range(nodes)),
        lat,
        lon,
        elevation,
        edge_index,
        edge_attr,
        source="TEST_FIXTURE_ONLY",
    )


def test_missing_rh_has_explicit_availability_mask():
    ds = dataset()
    split = ("2023-01-02T00", "2023-01-10T23")
    norm = fit_normalizer(ds, split)
    dynamic = normalized_dynamic(ds, norm)
    assert dynamic.shape[-1] == 3
    assert np.all(dynamic[..., 2] == 0.0)
    assert np.isfinite(dynamic).all()


def test_source_bound_uses_training_labels_only():
    ds = dataset()
    split = ("2023-01-02T00", "2023-01-10T23")
    norm = fit_normalizer(ds, split)
    before = derive_source_bound(ds, norm, split, quantile=0.99)
    changed = ds.temperature.copy()
    changed[300:] += 1000.0
    contaminated = UrbanDataset(
        ds.name,
        ds.timestamps,
        changed,
        ds.rh,
        ds.observed_mask,
        ds.station_ids,
        ds.lat,
        ds.lon,
        ds.elevation,
        ds.edge_index,
        ds.edge_attr,
        ds.source,
    )
    after = derive_source_bound(contaminated, norm, split, quantile=0.99)
    assert before == after
