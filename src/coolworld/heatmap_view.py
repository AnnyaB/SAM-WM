from __future__ import annotations

from copy import deepcopy
from typing import Any


class HeatmapSchemaError(ValueError):
    pass


def validated_heatmap_feature_collection(map_data: dict[str, Any]) -> dict[str, Any]:
    """Validate documented FortyGuard TCM fields used by the 3D renderer.

    No temperature is generated or imputed. A tile without a real numeric
    `average_temperature` is rejected from the thermal 3D view.
    """
    if map_data.get("type") != "FeatureCollection":
        raise HeatmapSchemaError("map_data is not a GeoJSON FeatureCollection")
    features = map_data.get("features")
    if not isinstance(features, list) or not features:
        raise HeatmapSchemaError("map_data contains no heatmap features")
    out = deepcopy(map_data)
    for feature in out["features"]:
        props = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(props, dict) or not isinstance(geometry, dict):
            raise HeatmapSchemaError("heatmap feature lacks properties/geometry")
        value = props.get("average_temperature")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HeatmapSchemaError("heatmap tile lacks numeric average_temperature")
        props["cw_observed_temperature_c"] = float(value)
    return out
