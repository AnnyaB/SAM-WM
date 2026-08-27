from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .deployment import (
    DeploymentError,
    baseline_forecast,
    canonical_map_data,
    counterfactual_forecast,
    grid_signature,
    validate_deployment_bundle,
)
from .experiment import load_checkpoint
from .fortyguard import FortyGuardClient
from .provider import recorded_heatmap_frames, validate_provider_replay

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "static"
EVIDENCE_DIR = Path(os.getenv("COOLWORLD_EVIDENCE_DIR", "artifacts/fortyguard"))
CHECKPOINT = Path(os.getenv("SAMWM_CHECKPOINT", "artifacts/deployment/best.pt"))
CALIBRATION = Path(os.getenv("SAMWM_CALIBRATION", "artifacts/deployment/calibration.json"))
EVALUATION = Path(os.getenv("SAMWM_EVALUATION", "artifacts/deployment/evaluation.json"))
PROVIDER_REPLAY = Path(
    os.getenv("SAMWM_PROVIDER_REPLAY", "artifacts/deployment/fortyguard_replay.json")
)
CANDRA_ACTIONS = Path(
    os.getenv(
        "SAMWM_CANDRA_ACTIONS",
        os.getenv("SAMWM_CANDRA_EFFECT", "artifacts/deployment/candra_actions.json"),
    )
)

app = FastAPI(
    title="SAM-WM · CoolWorld",
    version="1.1.0",
    description="Evidence-bounded urban thermal world-model interface",
)
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("data", {}).get("result", {})
    if not isinstance(result, dict):
        raise ValueError("FortyGuard result is not an object")
    return result


def _map_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = _result(payload).get("map_data")
    if not isinstance(data, dict):
        raise ValueError("FortyGuard result does not contain GeoJSON map_data")
    try:
        return canonical_map_data(data)
    except DeploymentError as exc:
        raise ValueError(str(exc)) from exc


def _timeline(limit: int) -> list[dict[str, Any]]:
    return recorded_heatmap_frames(EVIDENCE_DIR, limit=limit)


def _has_consecutive_hourly_context(frames: list[dict[str, Any]], count: int) -> bool:
    if len(frames) < count:
        return False
    selected = frames[-count:]
    try:
        stamps = [datetime.fromisoformat(str(frame["timestamp"])) for frame in selected]
    except (TypeError, ValueError):
        return False
    return all(
        (right - left).total_seconds() == 3600
        for left, right in zip(stamps, stamps[1:], strict=False)
    )


def _deployment_state(frames: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint_sha: str | None = None
    context_hours = 48
    bundle_ready = False
    replay_ready = False
    bundle_status = "MODEL_NOT_READY"
    replay_status = "FORTYGUARD_REPLAY_REQUIRED"

    try:
        bundle = validate_deployment_bundle(CHECKPOINT, CALIBRATION, EVALUATION)
        checkpoint_sha = bundle.checkpoint_sha256
        bundle_ready = True
        bundle_status = "DEPLOYMENT_BUNDLE_VALID"
        _, _, cfg, _ = load_checkpoint(CHECKPOINT)
        context_hours = int(cfg["context_hours"])
        validate_provider_replay(PROVIDER_REPLAY, checkpoint_sha)
        replay_ready = True
        replay_status = "FORTYGUARD_REPLAY_VALID"
    except DeploymentError as exc:
        if bundle_ready:
            replay_status = str(exc)
        else:
            bundle_status = str(exc)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        bundle_status = f"DEPLOYMENT_BUNDLE_INVALID: {exc}"

    context_ready = _has_consecutive_hourly_context(frames, context_hours)
    candra_ready = CANDRA_ACTIONS.is_file()
    forecast_ready = bundle_ready and replay_ready and context_ready
    counterfactual_ready = forecast_ready and candra_ready

    if counterfactual_ready:
        status = "READY"
    elif not bundle_ready:
        status = bundle_status
    elif not replay_ready:
        status = replay_status
    elif not context_ready:
        status = f"REAL_CONTEXT_REQUIRES_{context_hours}_CONSECUTIVE_HOURS"
    else:
        status = "FORECAST_READY_CANDRA_ACTION_EVIDENCE_REQUIRED"

    return {
        "ready": counterfactual_ready,
        "forecast_ready": forecast_ready,
        "status": status,
        "model_id": CHECKPOINT.name if CHECKPOINT.is_file() else None,
        "checkpoint_sha256": checkpoint_sha,
        "calibration_ready": bundle_ready,
        "evaluation_ready": bundle_ready,
        "provider_replay_ready": replay_ready,
        "provider_replay_status": replay_status,
        "context_hours_required": context_hours,
        "context_bundle_ready": context_ready,
        "context_manifest_ready": bool(frames),
        "candra_effect_ready": candra_ready,
        "engine_promoted": forecast_ready,
    }


@app.get("/")
def index() -> FileResponse:
    if not (STATIC / "index.html").exists():
        raise HTTPException(status_code=503, detail="UI_NOT_INSTALLED")
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "evidence_policy": "real_only_fail_closed",
        "fortyguard_key_configured": bool(os.getenv("FORTYGUARD_API_KEY")),
    }


@app.get("/api/readiness")
def readiness() -> dict[str, Any]:
    frames = _timeline(1000)
    model = _deployment_state(frames)
    return {
        "evidence_policy": "real_only_fail_closed",
        "fortyguard_key_configured": bool(os.getenv("FORTYGUARD_API_KEY")),
        "counterfactual_model": model,
        "recorded_real_frames": len(frames),
        "world_modes": {
            "observed": "real FortyGuard evidence",
            "validated_replay": "immutable provider/intervention evidence only",
            "predicted_future": (
                "frozen SAM-WM output only after final/OOD bundle + provider replay + real context"
            ),
            "counterfactual": "requires independent CANDRA action evidence in addition to forecast",
        },
    }


@app.get("/api/evidence/timeline")
def evidence_timeline(limit: int = 96) -> dict[str, Any]:
    frames = _timeline(limit)
    return {
        "mode": "observed",
        "frame_count": len(frames),
        "grid_signature": frames[-1]["grid_signature"] if frames else None,
        "frames": frames,
    }


@app.post("/api/fortyguard/heatmap")
def fortyguard_heatmap(payload: dict[str, Any]) -> dict[str, Any]:
    if not os.getenv("FORTYGUARD_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured",
        )
    try:
        result = FortyGuardClient(EVIDENCE_DIR).heatmap(payload)
        map_data = _map_data(result.payload)
        stats = _result(result.payload).get("stats_data")
    except (RuntimeError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    return {
        "activity_id": result.activity_id,
        "mode": "observed",
        "label": "OBSERVED FORTYGUARD THERMAL FIELD",
        "provenance": {
            "request_sha256": result.request_sha256,
            "content_sha256": result.content_sha256,
        },
        "map_data": map_data,
        "grid_signature": grid_signature(map_data),
        "stats_data": stats,
    }


@app.post("/api/forecast")
def forecast(payload: dict[str, Any]) -> dict[str, Any]:
    frames = _timeline(1000)
    state = _deployment_state(frames)
    if not state["forecast_ready"]:
        raise HTTPException(status_code=409, detail=state["status"])
    requested_signature = payload.get("grid_signature")
    if requested_signature and requested_signature != frames[-1]["grid_signature"]:
        raise HTTPException(status_code=409, detail="REQUEST_GRID_DOES_NOT_MATCH_REAL_CONTEXT")
    try:
        prediction = baseline_forecast(CHECKPOINT, CALIBRATION, EVALUATION, frames)
    except DeploymentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "mode": "predicted",
        "label": "SAM-WM MODEL PREDICTION — NOT OBSERVED",
        "prediction": prediction,
    }


@app.post("/api/counterfactual")
def counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    frames = _timeline(1000)
    state = _deployment_state(frames)
    if not state["ready"]:
        raise HTTPException(status_code=409, detail=state["status"])
    requested_signature = payload.get("grid_signature")
    if requested_signature != frames[-1]["grid_signature"]:
        raise HTTPException(status_code=409, detail="REQUEST_GRID_DOES_NOT_MATCH_REAL_CONTEXT")

    kind = str(payload.get("kind", "")).strip()
    tile_ids = payload.get("tile_ids")
    try:
        coverage_fraction = float(payload.get("coverage_fraction"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="ACTION_COVERAGE_INVALID") from exc
    if not kind or not isinstance(tile_ids, list):
        raise HTTPException(status_code=422, detail="ACTION_REQUEST_INVALID")

    try:
        prediction = counterfactual_forecast(
            CHECKPOINT,
            CALIBRATION,
            EVALUATION,
            CANDRA_ACTIONS,
            frames,
            kind=kind,
            coverage_fraction=coverage_fraction,
            tile_ids=[str(value) for value in tile_ids],
        )
    except DeploymentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "mode": "predicted_counterfactual",
        "label": "SUPPORT-GATED MODEL PREDICTION — NOT OBSERVED",
        "prediction": prediction,
    }
