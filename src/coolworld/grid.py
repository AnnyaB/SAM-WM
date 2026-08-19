from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from typing import Any


def _polygon_vertex_mean(coords: list[Any]) -> tuple[float, float]:
    ring = coords[0]
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    if not points:
        raise ValueError("polygon has no coordinates")
    return (
        sum(float(p[0]) for p in points) / len(points),
        sum(float(p[1]) for p in points) / len(points),
    )


def grid_signature_from_geojson(map_data: dict[str, Any]) -> str:
    if map_data.get("type") != "FeatureCollection":
        raise ValueError("grid signature requires FeatureCollection")
    rows: list[tuple[str, float, float]] = []
    for feature in map_data.get("features", []):
        props = feature.get("properties", {})
        tile_id = str(props.get("tile_id", feature.get("id", "")))
        geom = feature.get("geometry", {})
        if not tile_id or geom.get("type") != "Polygon":
            raise ValueError("every grid feature needs tile_id and Polygon geometry")
        lon, lat = _polygon_vertex_mean(geom["coordinates"])
        rows.append((tile_id, round(lon, 7), round(lat, 7)))
    if not rows:
        raise ValueError("grid is empty")
    raw = json.dumps(sorted(rows), separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(raw).hexdigest()


def grid_signature_from_centroids(rows: Iterable[tuple[str, float, float]]) -> str:
    normalized = sorted(
        (str(tile_id), round(float(lon), 7), round(float(lat), 7)) for tile_id, lon, lat in rows
    )
    if not normalized:
        raise ValueError("grid is empty")
    raw = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(raw).hexdigest()
