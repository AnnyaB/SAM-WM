from __future__ import annotations

import math

import numpy as np
import torch


def haversine_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """City-centred equirectangular coordinates in metres.

    The approximation is accurate for the compact urban sensor/AOI extents used by
    SAM-WM and, unlike raw latitude/longitude, does not encode city identity.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 1 or len(lat) == 0:
        raise ValueError("lat/lon must be non-empty aligned 1-D arrays")
    lat0 = np.deg2rad(float(np.mean(lat)))
    x = np.deg2rad(lon - np.mean(lon)) * 6_371_000.0 * math.cos(lat0)
    y = np.deg2rad(lat - np.mean(lat)) * 6_371_000.0
    return x, y


def local_static_features(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """Translation-invariant geometry features for zero-shot cross-city transfer.

    x/y are normalised by the city's RMS radius (geometry only, no target labels).
    Elevation is expressed relative to the city's median and scaled by a robust
    geometry-only scale. The representation therefore keeps local shape/topography
    without turning absolute latitude/longitude into a city-ID shortcut.
    """
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
    """Undirected sparse kNN graph with direction and physical edge length.

    edge_attr = [unit_x, unit_y, log1p(distance_km)]. The first two channels are
    retained for wind projection; distance gives exchange/message operators a
    physically meaningful locality cue without dense O(N^2) attention.
    """
    if len(lat) < 2:
        raise ValueError("at least two stations required")
    if k < 1:
        raise ValueError("k must be >=1")
    x, y = haversine_xy(np.asarray(lat, float), np.asarray(lon, float))
    pts = np.column_stack([x, y])
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    pairs: set[tuple[int, int]] = set()
    kk = min(k, len(lat) - 1)
    for i in range(len(lat)):
        for j in np.argsort(d[i])[1 : kk + 1]:
            a, b = sorted((i, int(j)))
            pairs.add((a, b))
    pairs_sorted = sorted(pairs)
    edge_index = torch.tensor(pairs_sorted, dtype=torch.long).t().contiguous()
    attrs = []
    for i, j in pairs_sorted:
        vec = pts[j] - pts[i]
        dist_m = max(float(np.linalg.norm(vec)), 1.0)
        unit = vec / dist_m
        attrs.append([float(unit[0]), float(unit[1]), math.log1p(dist_m / 1000.0)])
    return edge_index, torch.tensor(attrs, dtype=torch.float32)
