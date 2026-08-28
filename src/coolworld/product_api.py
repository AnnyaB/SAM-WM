from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from .deployment import (
    DeploymentError,
    baseline_forecast,
    canonical_map_data,
    validate_deployment_bundle,
)
from .experiment import load_checkpoint
from .provider import recorded_heatmap_frames, validate_provider_replay

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = Path(os.getenv("COOLWORLD_EVIDENCE_DIR", "artifacts/fortyguard"))
CHECKPOINT = Path(os.getenv("SAMWM_CHECKPOINT", "artifacts/deployment/best.pt"))
CALIBRATION = Path(os.getenv("SAMWM_CALIBRATION", "artifacts/deployment/calibration.json"))
EVALUATION = Path(os.getenv("SAMWM_EVALUATION", "artifacts/deployment/evaluation.json"))
PROVIDER_REPLAY = Path(
    os.getenv("SAMWM_PROVIDER_REPLAY", "artifacts/deployment/fortyguard_replay.json")
)
PROMOTION_MANIFEST = Path(
    os.getenv("SAMWM_PROMOTION_MANIFEST", "artifacts/deployment/PROMOTION_MANIFEST.json")
)
SUMMARY = Path(os.getenv("SAMWM_SUMMARY", "artifacts/summary.json"))
CANDRA_ACTIONS = Path(
    os.getenv(
        "SAMWM_CANDRA_ACTIONS",
        os.getenv("SAMWM_CANDRA_EFFECT", "artifacts/deployment/candra_actions.json"),
    )
)

router = APIRouter()

_FORECAST_CACHE_LOCK = threading.RLock()
_FORECAST_CACHE_KEY: str | None = None
_FORECAST_CACHE_VALUE: dict[str, Any] | None = None


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _timeline(limit: int = 1000) -> list[dict[str, Any]]:
    return recorded_heatmap_frames(EVIDENCE_DIR, limit=limit)


def _has_consecutive_hourly_context(frames: list[dict[str, Any]], count: int) -> bool:
    if len(frames) < count:
        return False
    selected = frames[-count:]
    try:
        stamps = [datetime.fromisoformat(str(frame["timestamp"])) for frame in selected]
    except (KeyError, TypeError, ValueError):
        return False
    return all(
        (right - left).total_seconds() == 3600
        for left, right in zip(stamps, stamps[1:], strict=False)
    )


def _promotion_state(checkpoint_sha: str | None) -> tuple[bool, str, dict[str, Any] | None]:
    if not checkpoint_sha:
        return False, "PROMOTED_CHECKPOINT_UNAVAILABLE", None

    payload = _read_json(PROMOTION_MANIFEST)
    if payload is None:
        return False, "PROMOTION_MANIFEST_REQUIRED", None
    if payload.get("protocol") != "SAM_WM_DEPLOYMENT_PROMOTION_V1":
        return False, "INVALID_PROMOTION_PROTOCOL", payload
    if payload.get("checkpoint_sha256") != checkpoint_sha:
        return False, "PROMOTION_CHECKPOINT_HASH_MISMATCH", payload
    if payload.get("model") != "SAM-WM":
        return False, "PROMOTION_MODEL_MISMATCH", payload
    return True, "PROMOTED_MODEL_BUNDLE_VALID", payload


def _provider_replay_snapshot(checkpoint_sha: str | None) -> dict[str, Any]:
    payload = _read_json(PROVIDER_REPLAY)
    if payload is None:
        return {
            "present": False,
            "status": "MISSING",
            "reason": "FORTYGUARD_REPLAY_REQUIRED",
        }

    return {
        "present": True,
        "status": str(payload.get("status", "UNKNOWN")),
        "protocol": payload.get("protocol"),
        "checkpoint_matches": bool(
            checkpoint_sha and payload.get("checkpoint_sha256") == checkpoint_sha
        ),
        "conformal_coverage": payload.get("conformal_coverage"),
        "minimum_required_coverage": payload.get("minimum_required_coverage"),
        "mae_to_radius_ratio": payload.get("mae_to_radius_ratio"),
        "maximum_allowed_mae_to_radius_ratio": payload.get("maximum_allowed_mae_to_radius_ratio"),
        "model_mae_c": payload.get("model_mae_c"),
        "model_rmse_c": payload.get("model_rmse_c"),
        "window_count": payload.get("window_count"),
        "frame_count": payload.get("frame_count"),
        "claim_boundary": payload.get("claim_boundary"),
    }


def product_state() -> dict[str, Any]:
    frames = _timeline()
    checkpoint_sha: str | None = None
    context_hours = 48
    bundle_ready = False
    bundle_reason = "MODEL_NOT_READY"

    try:
        bundle = validate_deployment_bundle(CHECKPOINT, CALIBRATION, EVALUATION)
        checkpoint_sha = bundle.checkpoint_sha256
        bundle_ready = True
        bundle_reason = "DEPLOYMENT_BUNDLE_VALID"
        _, _, cfg, _ = load_checkpoint(CHECKPOINT)
        context_hours = int(cfg["context_hours"])
    except DeploymentError as exc:
        bundle_reason = str(exc)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        bundle_reason = f"DEPLOYMENT_BUNDLE_INVALID: {exc}"

    promotion_ready, promotion_reason, promotion = _promotion_state(checkpoint_sha)
    context_ready = _has_consecutive_hourly_context(frames, context_hours)

    operational_certified = False
    operational_reason = "FORTYGUARD_REPLAY_REQUIRED"
    if checkpoint_sha:
        try:
            validate_provider_replay(PROVIDER_REPLAY, checkpoint_sha)
            operational_certified = True
            operational_reason = "FORTYGUARD_REPLAY_VALID"
        except DeploymentError as exc:
            operational_reason = str(exc)

    research_forecast_ready = bundle_ready and promotion_ready and context_ready
    candra_ready = CANDRA_ACTIONS.is_file()
    causal_action_ready = research_forecast_ready and operational_certified and candra_ready

    return {
        "real_provider_evidence_ready": bool(frames),
        "recorded_real_frames": len(frames),
        "context_hours_required": context_hours,
        "context_ready": context_ready,
        "model_bundle_valid": bundle_ready,
        "model_bundle_reason": bundle_reason,
        "model_bundle_promoted": promotion_ready,
        "promotion_reason": promotion_reason,
        "selected_seed": promotion.get("selected_seed") if promotion else None,
        "checkpoint_sha256": checkpoint_sha,
        "research_forecast_ready": research_forecast_ready,
        "operational_certified": operational_certified,
        "operational_certification_reason": operational_reason,
        "provider_replay": _provider_replay_snapshot(checkpoint_sha),
        "causal_action_ready": causal_action_ready,
        "causal_action_reason": (
            "CANDRA_ACTION_EVIDENCE_PRESENT"
            if candra_ready
            else "INDEPENDENT_TREATED_CONTROL_ACTION_EVIDENCE_REQUIRED"
        ),
        "live_provider_api_enabled": env_flag("COOLWORLD_LIVE_API_ENABLED"),
        "fortyguard_key_configured": bool(os.getenv("FORTYGUARD_API_KEY")),
    }


def _forecast_cache_key(frames: list[dict[str, Any]]) -> str:
    identity = {
        "checkpoint_mtime_ns": CHECKPOINT.stat().st_mtime_ns if CHECKPOINT.is_file() else None,
        "calibration_mtime_ns": CALIBRATION.stat().st_mtime_ns if CALIBRATION.is_file() else None,
        "evaluation_mtime_ns": EVALUATION.stat().st_mtime_ns if EVALUATION.is_file() else None,
        "frames": [
            {
                "timestamp": frame.get("timestamp"),
                "content_sha256": frame.get("content_sha256"),
                "grid_signature": frame.get("grid_signature"),
            }
            for frame in frames[-64:]
        ],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_forecast(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the exact frozen model with a one-entry process-local cache."""
    global _FORECAST_CACHE_KEY, _FORECAST_CACHE_VALUE

    key = _forecast_cache_key(frames)
    with _FORECAST_CACHE_LOCK:
        if key == _FORECAST_CACHE_KEY and _FORECAST_CACHE_VALUE is not None:
            return _FORECAST_CACHE_VALUE

    prediction = baseline_forecast(CHECKPOINT, CALIBRATION, EVALUATION, frames)

    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE_KEY = key
        _FORECAST_CACHE_VALUE = prediction

    return prediction


def _geometry_points(value: Any, out: list[tuple[float, float]]) -> None:
    if isinstance(value, (list, tuple)):
        if (
            len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            out.append((float(value[0]), float(value[1])))
            return
        for child in value:
            _geometry_points(child, out)


def _feature_center(feature: dict[str, Any]) -> dict[str, float] | None:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None

    points: list[tuple[float, float]] = []
    _geometry_points(geometry.get("coordinates"), points)
    if not points:
        return None

    return {
        "lon": 0.5 * (min(point[0] for point in points) + max(point[0] for point in points)),
        "lat": 0.5 * (min(point[1] for point in points) + max(point[1] for point in points)),
    }


def _metric(summary: dict[str, Any], evaluation: str, metric: str) -> dict[str, Any] | None:
    value = summary.get(evaluation, {}).get("metrics", {}).get(metric)
    return value if isinstance(value, dict) else None


def _benchmark_summary() -> dict[str, Any]:
    summary = _read_json(SUMMARY) or {}
    mapping = {
        "freiburg_id": "freiburg_heldout",
        "novi_sad_ood": "novisad_heldout",
        "turku_fairurbtemp_ood": "fairurbtemp_heldout",
    }
    output: dict[str, Any] = {}
    for label, evaluation in mapping.items():
        output[label] = {
            "mae_c": _metric(summary, evaluation, "mae"),
            "rmse_c": _metric(summary, evaluation, "rmse"),
            "conformal_coverage": _metric(summary, evaluation, "conformal_coverage"),
            "latency_ms_per_window": _metric(summary, evaluation, "latency_ms_per_window"),
            "parameter_count": _metric(summary, evaluation, "parameter_count"),
        }
    return output


@router.get("/api/product-status")
def product_status() -> dict[str, Any]:
    """Judge/user-facing readiness semantics without collapsing distinct gates."""
    return product_state()


@router.get("/api/evidence-summary")
def evidence_summary() -> dict[str, Any]:
    state = product_state()
    return {
        "model": "SAM-WM",
        "checkpoint_sha256": state["checkpoint_sha256"],
        "selected_seed": state["selected_seed"],
        "recorded_real_frames": state["recorded_real_frames"],
        "benchmarks": _benchmark_summary(),
        "provider_replay": state["provider_replay"],
        "claim_boundary": {
            "supported": [
                "real-data urban temperature forecasting",
                "multi-step +1…+6 h forecasting",
                "cross-city zero-shot evaluation",
                "uncertainty-aware research forecasting",
                "future-hotspot prioritization as non-causal decision support",
            ],
            "not_supported": [
                "causal cooling magnitude without treated/control evidence",
                "operational certification when the fixed replay gate fails",
                "planetary-scale cooling validation",
                "human-child-level general intelligence",
                "universal SOTA superiority",
            ],
        },
    }


@router.get("/api/hotspots")
def hotspots(fraction: float = 0.20) -> dict[str, Any]:
    if not 0.05 <= float(fraction) <= 0.50:
        raise HTTPException(status_code=422, detail="HOTSPOT_FRACTION_MUST_BE_0.05_TO_0.50")

    frames = _timeline()
    state = product_state()
    if not state["research_forecast_ready"]:
        raise HTTPException(status_code=409, detail="RESEARCH_FORECAST_NOT_READY")

    try:
        prediction = frozen_forecast(frames)
    except DeploymentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    tile_ids = [str(value) for value in prediction.get("tile_ids", [])]
    matrix = prediction.get("baseline_temperature_c")
    if not tile_ids or not isinstance(matrix, list) or not matrix:
        raise HTTPException(status_code=409, detail="MODEL_FORECAST_EMPTY")

    rows_by_horizon: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(tile_ids):
            raise HTTPException(status_code=409, detail="MODEL_FORECAST_SHAPE_INVALID")
        numeric = [float(value) for value in row]
        if not all(math.isfinite(value) for value in numeric):
            raise HTTPException(status_code=409, detail="MODEL_FORECAST_NONFINITE")
        rows_by_horizon.append(numeric)

    latest = canonical_map_data(frames[-1]["map_data"])
    latest_by_id = {
        str(feature["properties"]["tile_id"]): feature for feature in latest["features"]
    }

    future_by_tile = [[row[index] for row in rows_by_horizon] for index in range(len(tile_ids))]
    future_mean = [sum(series) / len(series) for series in future_by_tile]

    order = sorted(
        range(len(tile_ids)),
        key=lambda index: future_mean[index],
        reverse=True,
    )
    count = max(1, math.ceil(len(order) * float(fraction)))

    top_count_per_horizon = max(1, math.ceil(len(tile_ids) * float(fraction)))
    top_sets: list[set[int]] = []
    for row in rows_by_horizon:
        top_sets.append(
            set(
                sorted(
                    range(len(tile_ids)),
                    key=lambda index: row[index],
                    reverse=True,
                )[:top_count_per_horizon]
            )
        )

    hotspots_out: list[dict[str, Any]] = []
    denominator = max(1, len(order) - 1)

    for rank, index in enumerate(order[:count], start=1):
        tile = tile_ids[index]
        feature = latest_by_id.get(tile)
        current_c: float | None = None
        center: dict[str, float] | None = None

        if feature is not None:
            raw = feature.get("properties", {}).get("cw_observed_temperature_c")
            try:
                value = float(raw)
                current_c = value if math.isfinite(value) else None
            except (TypeError, ValueError):
                current_c = None
            center = _feature_center(feature)

        series = future_by_tile[index]
        persistence = sum(index in top for top in top_sets) / len(top_sets)
        percentile = 1.0 - ((rank - 1) / denominator)

        hotspots_out.append(
            {
                "rank": rank,
                "tile_id": tile,
                "center": center,
                "current_temperature_c": current_c,
                "forecast_mean_c": future_mean[index],
                "forecast_max_c": max(series),
                "forecast_6h_c": series[-1],
                "delta_6h_vs_current_c": (
                    series[-1] - current_c if current_c is not None else None
                ),
                "future_hotspot_percentile": percentile,
                "top_fraction_persistence": persistence,
                "forecast_by_horizon_c": series,
            }
        )

    return {
        "mode": "research_hotspot_plan",
        "label": "SAM-WM FUTURE HOTSPOT PRIORITY — NON-CAUSAL DECISION SUPPORT",
        "actionable_cooling_effect": False,
        "fraction": float(fraction),
        "tile_count": len(tile_ids),
        "selected_count": len(hotspots_out),
        "conformal_radius_c": prediction.get("baseline_conformal_radius_c"),
        "checkpoint_sha256": prediction.get("checkpoint_sha256"),
        "context_sha256": prediction.get("context_sha256"),
        "future_timestamps": prediction.get("future_timestamps"),
        "hotspots": hotspots_out,
        "candidate_physical_interventions": [
            {
                "kind": "tree_canopy",
                "effect_c": None,
                "status": "REQUIRES_INDEPENDENT_CAUSAL_EVIDENCE",
            },
            {
                "kind": "shade",
                "effect_c": None,
                "status": "REQUIRES_INDEPENDENT_CAUSAL_EVIDENCE",
            },
            {
                "kind": "reflective_pavement",
                "effect_c": None,
                "status": "REQUIRES_INDEPENDENT_CAUSAL_EVIDENCE",
            },
        ],
        "claim_boundary": (
            "Hotspot ranking uses the frozen SAM-WM forecast only. "
            "It prioritizes where engineers may investigate physical cooling interventions; "
            "it does not estimate causal cooling."
        ),
    }
