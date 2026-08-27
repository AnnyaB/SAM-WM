from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .evidence import digest_json
from .fortyguard import FortyGuardClient, completed_heatmap_records

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "static"
EVIDENCE_DIR = Path(os.getenv("COOLWORLD_EVIDENCE_DIR", "artifacts/fortyguard"))
CHECKPOINT = Path(os.getenv("SAMWM_CHECKPOINT", "artifacts/deployment/best.pt"))
CALIBRATION = Path(os.getenv("SAMWM_CALIBRATION", "artifacts/deployment/calibration.json"))
CANDRA_EFFECT = Path(os.getenv("SAMWM_CANDRA_EFFECT", "artifacts/deployment/candra_effect.json"))

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
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("FortyGuard result does not contain GeoJSON map_data")
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("FortyGuard map_data contains no features")
    return data


def _timestamp(request: dict[str, Any]) -> str | None:
    date_time = request.get("date_time", {})
    date = date_time.get("start_date")
    clock = date_time.get("start_time")
    if not date or not clock:
        return None
    value = f"{date}T{clock}"
    return value if len(value) >= 16 else None


def _grid_signature(map_data: dict[str, Any]) -> str:
    geometry = [
        {
            "id": feature.get("properties", {}).get("tile_id", feature.get("id")),
            "geometry": feature.get("geometry"),
        }
        for feature in map_data.get("features", [])
    ]
    return digest_json(geometry)


def _timeline(limit: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for record in completed_heatmap_records(EVIDENCE_DIR):
        timestamp = _timestamp(record["request_payload"])
        if timestamp is None:
            continue
        try:
            map_data = _map_data(record["response"])
        except ValueError:
            continue
        frames.append(
            {
                "timestamp": timestamp,
                "activity_id": record["activity_id"],
                "request_sha256": record["request_sha256"],
                "content_sha256": record["content_sha256"],
                "grid_signature": _grid_signature(map_data),
                "map_data": map_data,
            }
        )
    frames.sort(key=lambda frame: frame["timestamp"])
    if not frames:
        return []
    latest_signature = frames[-1]["grid_signature"]
    compatible = [frame for frame in frames if frame["grid_signature"] == latest_signature]
    return compatible[-max(1, min(int(limit), 240)) :]


def _has_consecutive_hourly_context(frames: list[dict[str, Any]], count: int = 48) -> bool:
    if len(frames) < count:
        return False
    selected = frames[-count:]
    try:
        stamps = [datetime.fromisoformat(str(frame["timestamp"])) for frame in selected]
    except (TypeError, ValueError):
        return False
    return all((right - left).total_seconds() == 3600 for left, right in zip(stamps, stamps[1:]))


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
    frames = _timeline(240)
    checkpoint_ready = CHECKPOINT.exists()
    calibration_ready = CALIBRATION.exists()
    candra_ready = CANDRA_EFFECT.exists()
    context_ready = _has_consecutive_hourly_context(frames, 48)
    prerequisites_ready = checkpoint_ready and calibration_ready and candra_ready and context_ready

    if prerequisites_ready:
        status = "COUNTERFACTUAL_ENGINE_NOT_PROMOTED"
    elif checkpoint_ready and calibration_ready and context_ready:
        status = "FORECAST_READY_CANDRA_ACTION_EVIDENCE_REQUIRED"
    elif checkpoint_ready and calibration_ready:
        status = "FORECAST_READY_REAL_HOURLY_CONTEXT_REQUIRED"
    elif checkpoint_ready:
        status = "CHECKPOINT_READY_CALIBRATION_REQUIRED"
    else:
        status = "MODEL_NOT_READY"

    return {
        "evidence_policy": "real_only_fail_closed",
        "fortyguard_key_configured": bool(os.getenv("FORTYGUARD_API_KEY")),
        "counterfactual_model": {
            "ready": False,
            "status": status,
            "model_id": CHECKPOINT.name if checkpoint_ready else None,
            "checkpoint_sha256": None,
            "calibration_ready": calibration_ready,
            "context_bundle_ready": context_ready,
            "context_manifest_ready": bool(frames),
            "candra_effect_ready": candra_ready,
            "engine_promoted": False,
        },
        "recorded_real_frames": len(frames),
        "world_modes": {
            "observed": "real FortyGuard evidence",
            "validated_replay": "requires attributable treated/control intervention evidence",
            "counterfactual": "locked until a validated deployment inference engine is promoted",
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
        "grid_signature": _grid_signature(map_data),
        "stats_data": stats,
    }


@app.post("/api/counterfactual")
def counterfactual(_: dict[str, Any]) -> dict[str, Any]:
    state = readiness()["counterfactual_model"]
    raise HTTPException(status_code=409, detail=state["status"])
