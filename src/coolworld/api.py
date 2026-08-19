from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .evidence import EvidenceStore
from .fortyguard import (
    EnvironmentalParametersRequest,
    FortyGuardClient,
    FortyGuardError,
    HeatIntelligenceRequest,
    HeatmapRequest,
    SatelliteRequest,
    StreetViewRequest,
)
from .grid import grid_signature_from_geojson
from .heatmap_view import HeatmapSchemaError, validated_heatmap_feature_collection
from .interventions import Intervention, InterventionKind
from .ml.inference import CounterfactualInferenceEngine
from .model_gate import validate_counterfactual_artifact
from .phoenix import COOL_CORRIDORS, COOL_PAVEMENT, TREE_CANOPY, PhoenixOpenDataClient
from .timeline import load_recorded_heatmap_timeline

settings = Settings()
store = EvidenceStore(settings.evidence_dir)
phoenix = PhoenixOpenDataClient(settings, store)
model_artifact_dir = Path("artifacts/counterfactual_model")

app = FastAPI(
    title="CoolWorld-SAM",
    version="0.6.1",
    description="Real-evidence 3D urban heat intervention intelligence",
)

static_dir = Path(__file__).resolve().parents[2] / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "evidence_policy": "real_only",
        "fortyguard_key_configured": settings.has_fortyguard_key,
    }


@app.get("/api/evidence/timeline")
def evidence_timeline(limit: int = 48) -> dict[str, object]:
    frames = load_recorded_heatmap_timeline(settings.evidence_dir, limit=max(1, min(limit, 240)))
    return {
        "mode": "observed",
        "grid_signature": grid_signature_from_geojson(frames[-1].map_data) if frames else None,
        "frame_count": len(frames),
        "frames": [
            {
                "timestamp": f.timestamp,
                "activity_id": f.activity_id,
                "content_sha256": f.content_sha256,
                "request_sha256": f.request_sha256,
                "map_data": f.map_data,
            }
            for f in frames
        ],
    }


@app.get("/api/readiness")
def readiness() -> dict[str, object]:
    model = validate_counterfactual_artifact(model_artifact_dir)
    return {
        "evidence_policy": "real_only",
        "fortyguard_key_configured": settings.has_fortyguard_key,
        "counterfactual_model": {
            "ready": model.ready,
            "status": model.status,
            "model_id": model.model_id,
            "checkpoint_sha256": model.checkpoint_sha256,
            "calibration_ready": (model_artifact_dir / "support_calibration.json").exists(),
            "context_bundle_ready": settings.context_bundle.exists(),
            "context_manifest_ready": settings.context_manifest.exists(),
        },
        "world_modes": {
            "observed": "available when real source retrieval succeeds",
            "validated_replay": "requires saved real before/after intervention evaluation",
            "counterfactual": "requires validated evidence-bearing model artifact",
        },
    }


@app.get("/api/phoenix/cool-pavement")
def cool_pavement() -> dict:
    try:
        return phoenix.fetch_geojson(COOL_PAVEMENT)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CITY_DATA_UNAVAILABLE: {exc}") from exc


@app.get("/api/phoenix/cool-corridors")
def cool_corridors() -> dict:
    try:
        return phoenix.fetch_geojson(COOL_CORRIDORS)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CITY_DATA_UNAVAILABLE: {exc}") from exc


@app.get("/api/phoenix/tree-canopy")
def tree_canopy() -> dict:
    try:
        return phoenix.fetch_geojson(TREE_CANOPY)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CITY_DATA_UNAVAILABLE: {exc}") from exc


@app.post("/api/fortyguard/heatmap")
def fortyguard_heatmap(request: HeatmapRequest) -> dict:
    if not settings.has_fortyguard_key:
        raise HTTPException(
            status_code=503,
            detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured",
        )
    try:
        result = FortyGuardClient(settings, store).heatmap(request)
    except (FortyGuardError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"FORTYGUARD_UPSTREAM_ERROR: {exc}") from exc

    try:
        result_data = result.payload["data"]["result"]
        observed_map = validated_heatmap_feature_collection(result_data["map_data"])
    except (KeyError, TypeError, HeatmapSchemaError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"FORTYGUARD_SCHEMA_ERROR: {exc}",
        ) from exc

    return {
        "activity_id": result.activity_id,
        "mode": "observed",
        "label": "OBSERVED FORTYGUARD THERMAL FIELD",
        "provenance": {
            "kind": result.provenance.kind.value,
            "source_reference": result.provenance.source_reference,
            "retrieved_at_utc": result.provenance.retrieved_at_utc,
            "content_sha256": result.provenance.content_sha256,
            "request_sha256": result.provenance.request_sha256,
        },
        "map_data": observed_map,
        "grid_signature": grid_signature_from_geojson(observed_map),
        "stats_data": result_data.get("stats_data"),
    }


@app.post("/api/fortyguard/env-params")
def fortyguard_env_params(request: EnvironmentalParametersRequest) -> dict:
    if not settings.has_fortyguard_key:
        raise HTTPException(
            status_code=503, detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured"
        )
    try:
        result = FortyGuardClient(settings, store).env_params(request)
    except (FortyGuardError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    return {
        "activity_id": result.activity_id,
        "provenance": asdict(result.provenance),
        "result": result.payload.get("data", {}).get("result"),
    }


@app.post("/api/fortyguard/satellite")
def fortyguard_satellite(request: SatelliteRequest) -> dict:
    if not settings.has_fortyguard_key:
        raise HTTPException(
            status_code=503, detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured"
        )
    try:
        result = FortyGuardClient(settings, store).satellite(request)
    except (FortyGuardError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    return {
        "activity_id": result.activity_id,
        "provenance": asdict(result.provenance),
        "result": result.payload.get("data", {}).get("result"),
    }


@app.post("/api/fortyguard/streetview")
def fortyguard_streetview(request: StreetViewRequest) -> dict:
    if not settings.has_fortyguard_key:
        raise HTTPException(
            status_code=503, detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured"
        )
    try:
        result = FortyGuardClient(settings, store).streetview(request)
    except (FortyGuardError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    return {
        "activity_id": result.activity_id,
        "provenance": asdict(result.provenance),
        "result": result.payload.get("data", {}).get("result"),
    }


@app.post("/api/fortyguard/heat-intelligence")
def fortyguard_heat_intelligence(request: HeatIntelligenceRequest) -> dict:
    if not settings.has_fortyguard_key:
        raise HTTPException(
            status_code=503,
            detail="DATA_UNAVAILABLE: FORTYGUARD_API_KEY not configured",
        )
    try:
        activity_id, raw, provenance = FortyGuardClient(settings, store).heat_intelligence_pdf(
            request
        )
    except (FortyGuardError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"DATA_UNAVAILABLE: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"FORTYGUARD_UPSTREAM_ERROR: {exc}") from exc
    return {
        "activity_id": activity_id,
        "mode": "observed_analysis",
        "label": "FORTYGUARD HEAT INTELLIGENCE REPORT",
        "provenance": asdict(provenance),
        "pdf_bytes": len(raw),
        "content_sha256": provenance.content_sha256,
    }


class CounterfactualRequestBody(BaseModel):
    kind: InterventionKind
    grid_signature: str = Field(min_length=64, max_length=64)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    tile_ids: list[str]
    cost: float | None = None


@app.post("/api/counterfactual")
def counterfactual(request: CounterfactualRequestBody) -> dict[str, object]:
    model = validate_counterfactual_artifact(model_artifact_dir)
    if not model.ready:
        raise HTTPException(status_code=409, detail=model.status)
    if not settings.context_bundle.exists() or not settings.context_manifest.exists():
        raise HTTPException(status_code=409, detail="REAL_CONTEXT_DATA_NOT_READY")
    if not (model_artifact_dir / "support_calibration.json").exists():
        raise HTTPException(status_code=409, detail="SUPPORT_CALIBRATION_NOT_READY")
    try:
        intervention = Intervention(
            kind=request.kind,
            coverage_fraction=request.coverage_fraction,
            tile_ids=tuple(request.tile_ids),
            cost=request.cost,
        )
        engine = CounterfactualInferenceEngine(
            model_artifact_dir, settings.context_bundle, settings.context_manifest
        )
        result = engine.predict_latest(intervention, grid_signature=request.grid_signature)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=f"COUNTERFACTUAL_UNAVAILABLE: {exc}") from exc
    return {
        "mode": "counterfactual",
        "label": "MODEL PREDICTION — NOT OBSERVED",
        "intervention": {
            "kind": intervention.kind.value,
            "coverage_fraction": intervention.coverage_fraction,
            "tile_ids": list(intervention.tile_ids),
            "cost": intervention.cost,
        },
        "prediction": {
            "predicted_delta_c": result.predicted_delta_c,
            "interval_low_c": result.interval_low_c,
            "interval_high_c": result.interval_high_c,
            "support_score": result.support_score,
            "status": result.status,
            "horizon_delta_c": list(result.horizon_delta_c),
            "target_tile_ids": list(result.target_tile_ids),
            "target_horizon_delta_c": [list(row) for row in result.target_horizon_delta_c],
            "tile_ids": list(result.tile_ids),
            "future_timestamps": list(result.future_timestamps),
            "baseline_temperature_c": [list(row) for row in result.baseline_temperature_c],
            "candidate_temperature_c": [list(row) for row in result.candidate_temperature_c],
        },
    }
