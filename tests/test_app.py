from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import coolworld.app as app_module


def client() -> TestClient:
    return TestClient(app_module.app)


def test_ui_readiness_fails_closed_without_deployment_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(app_module, "CHECKPOINT", tmp_path / "missing.pt")
    monkeypatch.setattr(app_module, "CALIBRATION", tmp_path / "missing-calibration.json")
    monkeypatch.setattr(app_module, "CANDRA_EFFECT", tmp_path / "missing-candra.json")
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


def test_fortyguard_endpoint_requires_server_side_key(monkeypatch):
    monkeypatch.delenv("FORTYGUARD_API_KEY", raising=False)
    response = client().post("/api/fortyguard/heatmap", json={})
    assert response.status_code == 503
    assert "FORTYGUARD_API_KEY not configured" in response.json()["detail"]


def test_index_serves_the_real_ui():
    response = client().get("/")
    assert response.status_code == 200
    assert "CoolWorld" in response.text or "SAM-WM" in response.text
