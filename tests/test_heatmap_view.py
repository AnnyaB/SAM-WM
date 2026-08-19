import pytest

from coolworld.heatmap_view import HeatmapSchemaError, validated_heatmap_feature_collection


def test_heatmap_requires_average_temperature() -> None:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tile_id": 1},
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    }
    with pytest.raises(HeatmapSchemaError):
        validated_heatmap_feature_collection(data)


def test_heatmap_preserves_observed_temperature() -> None:
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tile_id": 1, "average_temperature": 41.25},
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    }
    out = validated_heatmap_feature_collection(data)
    assert out["features"][0]["properties"]["cw_observed_temperature_c"] == 41.25
