from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def haversine_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """City-centred equirectangular coordinates in metres.

    The approximation is accurate for compact urban sensor/AOI extents and,
    unlike raw latitude/longitude, does not encode city identity.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1 or len(lat) == 0:
        raise ValueError("lat/lon must be non-empty aligned 1-D arrays")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("lat/lon must be finite")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("lat/lon outside geographic bounds")

    lat0 = np.deg2rad(float(np.mean(lat)))
    x = np.deg2rad(lon - np.mean(lon)) * 6_371_000.0 * math.cos(lat0)
    y = np.deg2rad(lat - np.mean(lat)) * 6_371_000.0
    return x, y


def local_static_features(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """Translation-invariant geometry features for zero-shot cross-city transfer."""
    x, y = haversine_xy(lat, lon)
    radius = np.sqrt(x**2 + y**2)
    xy_scale = max(float(np.sqrt(np.mean(radius**2))), 100.0)
    x_rel = x / xy_scale
    y_rel = y / xy_scale

    elev = np.asarray(elevation, dtype=float)
    if elev.shape != x.shape:
        raise ValueError("elevation must align with lat/lon")
    finite = np.isfinite(elev)
    if finite.any():
        centre = float(np.nanmedian(elev))
        filled = np.where(finite, elev, centre)
        mad = float(np.nanmedian(np.abs(filled - centre)))
        scale = max(1.4826 * mad, 10.0)
        elev_rel = (filled - centre) / scale
    else:
        elev_rel = np.zeros_like(x)
    return np.column_stack([x_rel, y_rel, elev_rel]).astype(np.float32)


def knn_graph(lat: np.ndarray, lon: np.ndarray, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an undirected sparse physical kNN graph without dense N×N distances.

    `cKDTree` gives scalable neighbour discovery while the learned graph operators
    remain O(E). Edge attributes are `[unit_x, unit_y, log1p(distance_km)]`.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if len(lat) < 2:
        raise ValueError("at least two stations required")
    if k < 1:
        raise ValueError("k must be >=1")

    x, y = haversine_xy(lat, lon)
    pts = np.column_stack([x, y])
    kk = min(int(k), len(lat) - 1)

    tree = cKDTree(pts, compact_nodes=True, balanced_tree=True)
    query_k = min(len(lat), 2 * kk + 1)
    _, neighbours = tree.query(pts, k=query_k, workers=1)
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]

    pairs: set[tuple[int, int]] = set()
    for i, row in enumerate(neighbours):
        accepted = 0
        for raw_j in np.atleast_1d(row):
            j = int(raw_j)
            if j == i:
                continue
            a, b = sorted((i, j))
            pairs.add((a, b))
            accepted += 1
            if accepted == kk:
                break
        if accepted < kk:
            raise RuntimeError("kNN query returned insufficient non-self neighbours")

    pairs_sorted = sorted(pairs)
    edge_index = torch.tensor(pairs_sorted, dtype=torch.long).t().contiguous()
    attrs: list[list[float]] = []
    for i, j in pairs_sorted:
        dx = float(pts[j, 0] - pts[i, 0])
        dy = float(pts[j, 1] - pts[i, 1])
        dist_m = max(math.hypot(dx, dy), 1.0)
        attrs.append([dx / dist_m, dy / dist_m, math.log1p(dist_m / 1000.0)])
    return edge_index, torch.tensor(attrs, dtype=torch.float32)
