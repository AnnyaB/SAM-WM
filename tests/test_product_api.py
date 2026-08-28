from __future__ import annotations

from typing import Any

import coolworld.product_api as product_api


def square(tile_id: str, temperature: float, x0: float, y0: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {
            "tile_id": tile_id,
            "average_temperature": temperature,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [x0, y0],
                    [x0 + 0.001, y0],
                    [x0 + 0.001, y0 + 0.001],
                    [x0, y0 + 0.001],
                    [x0, y0],
                ]
            ],
        },
    }


def test_env_flag(monkeypatch):
    monkeypatch.delenv("COOLWORLD_LIVE_API_ENABLED", raising=False)
    assert product_api.env_flag("COOLWORLD_LIVE_API_ENABLED") is False

    monkeypatch.setenv("COOLWORLD_LIVE_API_ENABLED", "true")
    assert product_api.env_flag("COOLWORLD_LIVE_API_ENABLED") is True


def test_hotspot_plan_ranks_future_heat_without_inventing_causal_effect(monkeypatch):
    map_data = {
        "type": "FeatureCollection",
        "features": [
            square("a", 20.0, -121.90, 37.33),
            square("b", 21.0, -121.89, 37.33),
            square("c", 22.0, -121.88, 37.33),
            square("d", 23.0, -121.87, 37.33),
        ],
    }
    frames = [
        {
            "timestamp": "2026-08-22T00:00:00",
            "grid_signature": "grid",
            "map_data": map_data,
        }
    ]
    prediction = {
        "tile_ids": ["a", "b", "c", "d"],
        "future_timestamps": [
            "2026-08-22T01:00:00",
            "2026-08-22T02:00:00",
        ],
        "baseline_temperature_c": [
            [20.5, 22.0, 23.0, 24.0],
            [21.0, 22.5, 24.0, 25.0],
        ],
        "baseline_conformal_radius_c": 3.2,
        "checkpoint_sha256": "checkpoint",
        "context_sha256": "context",
    }

    monkeypatch.setattr(product_api, "_timeline", lambda limit=1000: frames)
    monkeypatch.setattr(
        product_api,
        "product_state",
        lambda: {"research_forecast_ready": True},
    )
    monkeypatch.setattr(product_api, "frozen_forecast", lambda _: prediction)

    result = product_api.hotspots(0.50)

    assert result["actionable_cooling_effect"] is False
    assert result["selected_count"] == 2
    assert result["hotspots"][0]["tile_id"] == "d"
    assert result["hotspots"][1]["tile_id"] == "c"
    assert result["candidate_physical_interventions"][0]["effect_c"] is None
    assert "does not estimate causal cooling" in result["claim_boundary"]


def test_hotspot_fraction_is_bounded():
    try:
        product_api.hotspots(0.01)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422
    else:
        raise AssertionError("Expected the hotspot fraction contract to fail closed")
