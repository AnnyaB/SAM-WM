from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import coolworld.provider as provider


def feature(tile_id: str, temperature: float, x: float) -> dict:
    return {
        "type": "Feature",
        "id": tile_id,
        "properties": {"average_temperature": temperature},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [x, 37.0],
                    [x + 0.001, 37.0],
                    [x + 0.001, 37.001],
                    [x, 37.001],
                    [x, 37.0],
                ]
            ],
        },
    }


def test_provider_replay_passes_only_with_real_outperformance_and_coverage(monkeypatch, tmp_path):
    context = 4
    horizon = 2
    total_frames = context + horizon + provider.MIN_REPLAY_WINDOWS - 1
    frames = []
    base = np.datetime64("2026-08-20T00:00:00")
    for index in range(total_frames):
        timestamp = np.datetime_as_string(base + index * np.timedelta64(1, "h"), unit="s")
        map_data = {
            "type": "FeatureCollection",
            "features": [
                feature("a", 30.0 + index, -122.0),
                feature("b", 31.0 + index, -121.99),
            ],
        }
        frames.append(
            {
                "timestamp": timestamp,
                "grid_signature": "same-grid",
                "map_data": map_data,
            }
        )

    monkeypatch.setattr(
        provider,
        "validate_deployment_bundle",
        lambda *args: SimpleNamespace(checkpoint_sha256="c" * 64),
    )
    monkeypatch.setattr(provider, "load_checkpoint", lambda *args: (None, None, {"context_hours": context, "horizon_hours": horizon}, None))

    def perfect_forecast(checkpoint, calibration, evaluation, context_frames):
        del checkpoint, calibration, evaluation
        last_index = int((np.datetime64(context_frames[-1]["timestamp"]) - base) / np.timedelta64(1, "h"))
        values = []
        timestamps = []
        for step in range(1, horizon + 1):
            idx = last_index + step
            values.append([30.0 + idx, 31.0 + idx])
            timestamps.append(np.datetime_as_string(base + idx * np.timedelta64(1, "h"), unit="s"))
        return {
            "tile_ids": ["a", "b"],
            "future_timestamps": timestamps,
            "baseline_temperature_c": values,
            "baseline_conformal_radius_c": 0.5,
        }

    monkeypatch.setattr(provider, "baseline_forecast", perfect_forecast)
    out = tmp_path / "replay.json"
    payload = provider.evaluate_provider_replay(
        tmp_path / "best.pt",
        tmp_path / "cal.json",
        tmp_path / "eval.json",
        frames,
        out,
    )
    assert payload["status"] == "PASS"
    assert payload["window_count"] == provider.MIN_REPLAY_WINDOWS
    assert payload["model_mae_c"] == 0.0
    assert payload["persistence_mae_c"] > 0.0
    assert payload["conformal_coverage"] == 1.0
    assert provider.validate_provider_replay(out, "c" * 64)["status"] == "PASS"
