from __future__ import annotations

import math

import numpy as np
import torch


def haversine_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = np.deg2rad(float(np.mean(lat)))
    x = np.deg2rad(lon - np.mean(lon)) * 6371000.0 * math.cos(lat0)
    y = np.deg2rad(lat - np.mean(lat)) * 6371000.0
    return x, y


def knn_graph(lat: np.ndarray, lon: np.ndarray, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    if len(lat) < 2:
        raise ValueError("at least two stations required")
    x, y = haversine_xy(lat.astype(float), lon.astype(float))
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
        dist = max(float(np.linalg.norm(vec)), 1.0)
        unit = vec / dist
        attrs.append([float(unit[0]), float(unit[1])])
    return edge_index, torch.tensor(attrs, dtype=torch.float32)
