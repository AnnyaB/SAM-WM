from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import coolworld.app as app_module


def client() -> TestClient:
    return TestClient(app_module.app)


def test_ui_readiness_fails_closed_without_deployment_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(app_module, "CHECKPOINT", tmp_path / "missing.pt")
    monkeypatch.setattr(app_module, "CALIBRATION", tmp_path / "missing-calibration.json")
    monkeypatch.setattr(app_module, "EVALUATION", tmp_path / "missing-evaluation.json")
    monkeypatch.setattr(app_module, "PROVIDER_REPLAY", tmp_path / "missing-replay.json")
    monkeypatch.setattr(app_module, "CANDRA_ACTIONS", tmp_path / "missing-candra.json")
    monkeypatch.setattr(app_module, "EVIDENCE_DIR", tmp_path / "evidence")

    response = client().get("/api/readiness")
    assert response.status_code == 200
    payload = response.json()
    model = payload["counterfactual_model"]
    assert payload["evidence_policy"] == "real_only_fail_closed"
    assert model["ready"] is False
    assert model["engine_promoted"] is False
    assert model["status"] == "MODEL_NOT_READY"

    counterfactual = client().post("/api/counterfactual", json={})
    assert counterfactual.status_code == 409
    assert counterfactual.json()["detail"] == "MODEL_NOT_READY"


def test_counterfactual_endpoint_reaches_promoted_engine(monkeypatch):
    frames = [{"timestamp": "2026-08-27T12:00:00", "grid_signature": "grid-1"}]
    monkeypatch.setattr(app_module, "_timeline", lambda _: frames)
    monkeypatch.setattr(
        app_module,
        "_deployment_state",
        lambda _: {"ready": True, "forecast_ready": True, "status": "READY"},
    )
    expected = {
        "future_timestamps": ["2026-08-27T13:00:00"],
        "tile_ids": ["tile-1"],
        "baseline_temperature_c": [[35.0]],
        "candidate_temperature_c": [[34.0]],
        "predicted_delta_c": -1.0,
        "interval_low_c": -1.2,
        "interval_high_c": -0.5,
        "support_score": 0.8,
        "status": "PREDICTED",
    }
    monkeypatch.setattr(app_module, "counterfactual_forecast", lambda *args, **kwargs: expected)

    response = client().post(
        "/api/counterfactual",
        json={
            "kind": "shade",
            "grid_signature": "grid-1",
            "coverage_fraction": 0.4,
            "tile_ids": ["tile-1"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "predicted_counterfactual"
    assert payload["prediction"] == expected


def test_fortyguard_endpoint_requires_server_side_key(monkeypatch):
    monkeypatch.delenv("FORTYGUARD_API_KEY", raising=False)
    response = client().post("/api/fortyguard/heatmap", json={})
    assert response.status_code == 503
    assert "FORTYGUARD_API_KEY not configured" in response.json()["detail"]


def test_index_serves_the_real_ui():
    response = client().get("/")
    assert response.status_code == 200
    assert "CoolWorld" in response.text or "SAM-WM" in response.text
