from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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


def replay_frames(context: int, horizon: int) -> tuple[list[dict], np.datetime64]:
    total_frames = context + horizon + provider.MIN_REPLAY_WINDOWS - 1
    frames = []
    base = np.datetime64("2026-08-20T00:00:00")
    for index in range(total_frames):
        timestamp = np.datetime_as_string(base + index * np.timedelta64(1, "h"), unit="s")
        frames.append(
            {
                "timestamp": timestamp,
                "grid_signature": "same-grid",
                "map_data": {
                    "type": "FeatureCollection",
                    "features": [
                        feature("a", 30.0 + index, -122.0),
                        feature("b", 31.0 + index, -121.99),
                    ],
                },
            }
        )
    return frames, base


def install_bundle_stubs(monkeypatch, context: int, horizon: int) -> None:
    monkeypatch.setattr(
        provider,
        "validate_deployment_bundle",
        lambda *args: SimpleNamespace(checkpoint_sha256="c" * 64),
    )
    monkeypatch.setattr(
        provider,
        "load_checkpoint",
        lambda *args: (None, None, {"context_hours": context, "horizon_hours": horizon}, None),
    )


def forecast_factory(base: np.datetime64, horizon: int, offset_c: float):
    def forecast(checkpoint, calibration, evaluation, context_frames):
        del checkpoint, calibration, evaluation
        last_index = int(
            (np.datetime64(context_frames[-1]["timestamp"]) - base) / np.timedelta64(1, "h")
        )
        values = []
        timestamps = []
        for step in range(1, horizon + 1):
            idx = last_index + step
            values.append([30.0 + idx + offset_c, 31.0 + idx + offset_c])
            timestamps.append(np.datetime_as_string(base + idx * np.timedelta64(1, "h"), unit="s"))
        return {
            "tile_ids": ["a", "b"],
            "future_timestamps": timestamps,
            "baseline_temperature_c": values,
            "baseline_conformal_radius_c": 0.5,
        }

    return forecast


def test_provider_replay_passes_same_samwm_when_coverage_and_error_gate_pass(monkeypatch, tmp_path):
    context = 4
    horizon = 2
    frames, base = replay_frames(context, horizon)
    install_bundle_stubs(monkeypatch, context, horizon)
    monkeypatch.setattr(provider, "baseline_forecast", forecast_factory(base, horizon, 0.0))

    out = tmp_path / "replay.json"
    payload = provider.evaluate_provider_replay(
        tmp_path / "best.pt",
        tmp_path / "cal.json",
        tmp_path / "eval.json",
        frames,
        out,
    )

    assert payload["protocol"] == "SAM_WM_FORTYGUARD_REPLAY_V2"
    assert payload["model"] == "SAM-WM"
    assert payload["status"] == "PASS"
    assert payload["window_count"] == provider.MIN_REPLAY_WINDOWS
    assert payload["model_mae_c"] == 0.0
    assert payload["mae_to_radius_ratio"] == 0.0
    assert payload["conformal_coverage"] == 1.0
    assert "persistence_mae_c" not in payload
    assert provider.validate_provider_replay(out, "c" * 64)["status"] == "PASS"


def test_provider_replay_fails_closed_when_same_samwm_error_exceeds_calibration(
    monkeypatch, tmp_path
):
    context = 4
    horizon = 2
    frames, base = replay_frames(context, horizon)
    install_bundle_stubs(monkeypatch, context, horizon)
    monkeypatch.setattr(provider, "baseline_forecast", forecast_factory(base, horizon, 1.0))

    out = tmp_path / "replay.json"
    payload = provider.evaluate_provider_replay(
        tmp_path / "best.pt",
        tmp_path / "cal.json",
        tmp_path / "eval.json",
        frames,
        out,
    )

    assert payload["status"] == "FAIL"
    assert payload["mae_to_radius_ratio"] > provider.MAX_MAE_TO_RADIUS_RATIO
    with pytest.raises(provider.DeploymentError, match="GATE_FAILED"):
        provider.validate_provider_replay(out, "c" * 64)
