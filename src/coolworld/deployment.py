from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .benchmarks import UrbanDataset
from .evidence import digest_json, sha256_file
from .experiment import load_checkpoint, normalized_dynamic, normalized_static
from .graph import knn_graph


class DeploymentError(RuntimeError):
    """Fail-closed deployment contract violation."""


@dataclass(frozen=True)
class DeploymentBundle:
    checkpoint_sha256: str
    conformal_radius_c: float
    evaluation: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"invalid deployment artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DeploymentError(f"deployment artifact is not an object: {path}")
    return payload


def validate_deployment_bundle(
    checkpoint: Path,
    calibration: Path,
    evaluation: Path,
) -> DeploymentBundle:
    """Validate promoted model, calibration and final/OOD evidence as one immutable bundle."""
    if not checkpoint.is_file():
        raise DeploymentError("MODEL_NOT_READY")
    if not calibration.is_file():
        raise DeploymentError("CALIBRATION_REQUIRED")
    if not evaluation.is_file():
        raise DeploymentError("FINAL_AND_OOD_EVIDENCE_REQUIRED")

    checkpoint_sha = sha256_file(checkpoint)
    cal = _load_json(calibration)
    if cal.get("protocol") != "SAM_WM_DEPLOYMENT_CALIBRATION_V1":
        raise DeploymentError("INVALID_CALIBRATION_PROTOCOL")
    if cal.get("checkpoint_sha256") != checkpoint_sha:
        raise DeploymentError("CALIBRATION_CHECKPOINT_HASH_MISMATCH")
    radius = float(cal.get("conformal_radius_c", float("nan")))
    if not np.isfinite(radius) or radius <= 0:
        raise DeploymentError("INVALID_CONFORMAL_RADIUS")

    evidence = _load_json(evaluation)
    if evidence.get("protocol") != "SAM_WM_DEPLOYMENT_EVIDENCE_V1":
        raise DeploymentError("INVALID_DEPLOYMENT_EVIDENCE_PROTOCOL")
    if evidence.get("checkpoint_sha256") != checkpoint_sha:
        raise DeploymentError("EVALUATION_CHECKPOINT_HASH_MISMATCH")
    required = evidence.get("required_evaluations")
    if not isinstance(required, dict):
        raise DeploymentError("EVALUATION_EVIDENCE_INCOMPLETE")
    if set(required) != {"freiburg_heldout", "novisad_heldout", "fairurbtemp_heldout"}:
        raise DeploymentError("EVALUATION_EVIDENCE_INCOMPLETE")
    if not all(isinstance(value, str) and len(value) == 64 for value in required.values()):
        raise DeploymentError("EVALUATION_EVIDENCE_HASH_INVALID")

    return DeploymentBundle(checkpoint_sha, radius, evidence)


def stable_tile_id(feature: dict[str, Any]) -> str:
    properties = feature.get("properties") or {}
    value = properties.get("tile_id", feature.get("id"))
    if value is not None and str(value).strip():
        return str(value)
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise DeploymentError("TILE_GEOMETRY_MISSING")
    return f"geo-{digest_json(geometry)[:24]}"


def _coordinates(value: Any, out: list[tuple[float, float]]) -> None:
    if isinstance(value, (list, tuple)):
        if (
            len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            out.append((float(value[0]), float(value[1])))
            return
        for child in value:
            _coordinates(child, out)


def _feature_center(feature: dict[str, Any]) -> tuple[float, float]:
    geometry = feature.get("geometry") or {}
    points: list[tuple[float, float]] = []
    _coordinates(geometry.get("coordinates"), points)
    if not points:
        raise DeploymentError("TILE_GEOMETRY_COORDINATES_MISSING")
    lon = 0.5 * (min(point[0] for point in points) + max(point[0] for point in points))
    lat = 0.5 * (min(point[1] for point in points) + max(point[1] for point in points))
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise DeploymentError("TILE_GEOMETRY_COORDINATES_INVALID")
    return lat, lon


def observed_temperature(feature: dict[str, Any]) -> float:
    properties = feature.get("properties") or {}
    raw = properties.get("cw_observed_temperature_c", properties.get("average_temperature"))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise DeploymentError("OBSERVED_TEMPERATURE_MISSING") from exc
    if not np.isfinite(value):
        raise DeploymentError("OBSERVED_TEMPERATURE_NONFINITE")
    return value


def canonical_map_data(map_data: dict[str, Any]) -> dict[str, Any]:
    """Validate real GeoJSON and attach stable tile IDs without changing observed temperatures."""
    if map_data.get("type") != "FeatureCollection":
        raise DeploymentError("MAP_DATA_NOT_FEATURE_COLLECTION")
    features = map_data.get("features")
    if not isinstance(features, list) or not features:
        raise DeploymentError("MAP_DATA_EMPTY")

    output: list[dict[str, Any]] = []
    ids: set[str] = set()
    for feature in features:
        if not isinstance(feature, dict):
            raise DeploymentError("MAP_FEATURE_INVALID")
        tile_id = stable_tile_id(feature)
        if tile_id in ids:
            raise DeploymentError("DUPLICATE_TILE_ID")
        ids.add(tile_id)
        observed_temperature(feature)
        _feature_center(feature)
        properties = dict(feature.get("properties") or {})
        properties["tile_id"] = tile_id
        properties["cw_observed_temperature_c"] = observed_temperature(feature)
        output.append({**feature, "properties": properties})
    return {**map_data, "features": output}


def grid_signature(map_data: dict[str, Any]) -> str:
    canonical = canonical_map_data(map_data)
    rows = sorted(
        (
            feature["properties"]["tile_id"],
            feature.get("geometry"),
        )
        for feature in canonical["features"]
    )
    return digest_json(rows)


def _context_dataset(frames: list[dict[str, Any]], *, k: int) -> UrbanDataset:
    if len(frames) < 2:
        raise DeploymentError("REAL_HOURLY_CONTEXT_REQUIRED")

    timestamps = np.asarray(
        [np.datetime64(str(frame["timestamp"]), "ns") for frame in frames],
        dtype="datetime64[ns]",
    )
    if np.isnat(timestamps).any() or not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise DeploymentError("CONTEXT_NOT_CONSECUTIVE_HOURLY")

    canonical_frames = [canonical_map_data(frame["map_data"]) for frame in frames]
    by_id = [
        {feature["properties"]["tile_id"]: feature for feature in frame["features"]}
        for frame in canonical_frames
    ]
    station_ids = tuple(sorted(by_id[-1]))
    if len(station_ids) < 2:
        raise DeploymentError("INSUFFICIENT_TILES_FOR_SPARSE_GRAPH")
    if any(set(frame) != set(station_ids) for frame in by_id):
        raise DeploymentError("CONTEXT_GRID_CHANGED")

    temperature = np.asarray(
        [[observed_temperature(frame[tile_id]) for tile_id in station_ids] for frame in by_id],
        dtype=np.float32,
    )
    if not np.isfinite(temperature).all():
        raise DeploymentError("CONTEXT_TEMPERATURE_INCOMPLETE")

    centers = [_feature_center(by_id[-1][tile_id]) for tile_id in station_ids]
    lat = np.asarray([center[0] for center in centers], dtype=np.float32)
    lon = np.asarray([center[1] for center in centers], dtype=np.float32)
    elevation = np.zeros(len(station_ids), dtype=np.float32)
    rh = np.full_like(temperature, np.nan, dtype=np.float32)
    observed = np.ones_like(temperature, dtype=bool)
    edge_index, edge_attr = knn_graph(lat, lon, min(int(k), len(station_ids) - 1))

    return UrbanDataset(
        name="fortyguard_live",
        timestamps=timestamps,
        temperature=temperature,
        rh=rh,
        observed_mask=observed,
        station_ids=station_ids,
        lat=lat,
        lon=lon,
        elevation=elevation,
        edge_index=edge_index,
        edge_attr=edge_attr,
        source="FortyGuard TCM recorded/live evidence",
    )


def _absolute_hours(timestamps: np.ndarray) -> np.ndarray:
    base = np.datetime64("2000-01-01T00:00:00")
    return ((timestamps.astype("datetime64[s]") - base) / np.timedelta64(1, "h")).astype(np.float32)


def baseline_forecast(
    checkpoint: Path,
    calibration: Path,
    evaluation: Path,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a frozen SAM-WM checkpoint zero-shot on a compatible real FortyGuard tile sequence."""
    bundle = validate_deployment_bundle(checkpoint, calibration, evaluation)
    device = torch.device("cpu")
    model, norm, cfg, _ = load_checkpoint(checkpoint, device=device)
    context = int(cfg["context_hours"])
    horizon = int(cfg["horizon_hours"])
    if len(frames) < context:
        raise DeploymentError(f"REAL_CONTEXT_REQUIRES_{context}_HOURS")
    selected = frames[-context:]
    ds = _context_dataset(selected, k=int(cfg.get("graph_k", 4)))

    dynamic = normalized_dynamic(ds, norm)
    static = normalized_static(ds)
    context_hours = _absolute_hours(ds.timestamps)
    future_timestamps = ds.timestamps[-1] + np.arange(1, horizon + 1) * np.timedelta64(1, "h")
    future_hours = _absolute_hours(future_timestamps)

    with torch.inference_mode():
        output = model(
            torch.from_numpy(dynamic).unsqueeze(0).to(device),
            torch.from_numpy(static).unsqueeze(0).to(device),
            torch.from_numpy(context_hours).unsqueeze(0).to(device),
            torch.from_numpy(future_hours).unsqueeze(0).to(device),
            ds.edge_index.to(device),
            ds.edge_attr.to(device),
        )
    baseline = output.temperature_mean[0].cpu().numpy() * norm.temp_std + norm.temp_mean
    if not np.isfinite(baseline).all():
        raise DeploymentError("MODEL_RETURNED_NONFINITE_FORECAST")

    radius = bundle.conformal_radius_c
    context_sha = digest_json(
        [
            {
                "timestamp": frame["timestamp"],
                "content_sha256": frame.get("content_sha256") or digest_json(frame["map_data"]),
            }
            for frame in selected
        ]
    )
    return {
        "tile_ids": list(ds.station_ids),
        "future_timestamps": [
            np.datetime_as_string(value, unit="s") for value in future_timestamps
        ],
        "baseline_temperature_c": baseline.tolist(),
        "baseline_interval_low_c": (baseline - radius).tolist(),
        "baseline_interval_high_c": (baseline + radius).tolist(),
        "baseline_conformal_radius_c": radius,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "context_sha256": context_sha,
        "evaluation_protocol": bundle.evaluation["protocol"],
        "context_modalities": {
            "temperature": "FortyGuard observed/provider thermal field",
            "relative_humidity": "unavailable; explicit availability mask=0",
            "elevation": "unavailable; relative elevation=0",
        },
    }


def _action_evidence(
    path: Path,
    *,
    kind: str,
    horizon: int,
    coverage_fraction: float,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise DeploymentError("CANDRA_ACTION_EVIDENCE_REQUIRED")
    payload = _load_json(path)
    if payload.get("protocol") != "SAM_WM_CANDRA_ACTIONS_V1":
        raise DeploymentError("INVALID_CANDRA_ACTION_PROTOCOL")
    actions = payload.get("actions")
    if not isinstance(actions, dict) or kind not in actions:
        raise DeploymentError("CANDRA_ACTION_NOT_SUPPORTED")
    action = actions[kind]
    if not isinstance(action, dict) or action.get("transfer_validated") is not True:
        raise DeploymentError("CANDRA_TRANSFER_NOT_VALIDATED")

    reference = float(action.get("reference_coverage_fraction", float("nan")))
    tolerance = float(action.get("coverage_tolerance", 0.0))
    if not (0 < reference <= 1 and 0 <= tolerance <= 1):
        raise DeploymentError("CANDRA_COVERAGE_CONTRACT_INVALID")
    if abs(float(coverage_fraction) - reference) > tolerance:
        raise DeploymentError("CANDRA_COVERAGE_OUT_OF_SUPPORT")

    for key in ("effect_c_by_horizon", "interval_low_c_by_horizon", "interval_high_c_by_horizon"):
        values = action.get(key)
        if not isinstance(values, list) or len(values) < horizon:
            raise DeploymentError("CANDRA_HORIZON_EVIDENCE_INCOMPLETE")
        array = np.asarray(values[:horizon], dtype=float)
        if not np.isfinite(array).all():
            raise DeploymentError("CANDRA_HORIZON_EVIDENCE_NONFINITE")

    effect = np.asarray(action["effect_c_by_horizon"][:horizon], dtype=float)
    low = np.asarray(action["interval_low_c_by_horizon"][:horizon], dtype=float)
    high = np.asarray(action["interval_high_c_by_horizon"][:horizon], dtype=float)
    if np.any(low > effect) or np.any(effect > high):
        raise DeploymentError("CANDRA_INTERVAL_ORDER_INVALID")
    support = float(action.get("support_score", float("nan")))
    if not 0 <= support <= 1:
        raise DeploymentError("CANDRA_SUPPORT_INVALID")
    if not action.get("source"):
        raise DeploymentError("CANDRA_SOURCE_PROVENANCE_REQUIRED")
    return action, sha256_file(path)


def counterfactual_forecast(
    checkpoint: Path,
    calibration: Path,
    evaluation: Path,
    candra_actions: Path,
    frames: list[dict[str, Any]],
    *,
    kind: str,
    coverage_fraction: float,
    tile_ids: list[str],
) -> dict[str, Any]:
    """Forecast baseline dynamics and apply only an independently supported action effect."""
    baseline = baseline_forecast(checkpoint, calibration, evaluation, frames)
    ids = baseline["tile_ids"]
    requested = list(dict.fromkeys(str(value) for value in tile_ids))
    if not requested:
        raise DeploymentError("ACTION_TILE_SELECTION_REQUIRED")
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise DeploymentError("ACTION_TILE_ID_NOT_IN_CONTEXT_GRID")
    if not 0 < float(coverage_fraction) <= 1:
        raise DeploymentError("ACTION_COVERAGE_INVALID")

    base = np.asarray(baseline["baseline_temperature_c"], dtype=float)
    horizon = base.shape[0]
    action, action_sha = _action_evidence(
        candra_actions,
        kind=kind,
        horizon=horizon,
        coverage_fraction=coverage_fraction,
    )
    effect = np.asarray(action["effect_c_by_horizon"][:horizon], dtype=float)
    low = np.asarray(action["interval_low_c_by_horizon"][:horizon], dtype=float)
    high = np.asarray(action["interval_high_c_by_horizon"][:horizon], dtype=float)
    index = {tile_id: position for position, tile_id in enumerate(ids)}
    selected = np.asarray([index[tile_id] for tile_id in requested], dtype=int)

    candidate = base.copy()
    candidate[:, selected] += effect[:, None]
    selected_fraction = len(selected) / len(ids)
    field_delta = candidate - base
    predicted_delta = float(field_delta.mean())
    interval_low = float((low * selected_fraction).mean())
    interval_high = float((high * selected_fraction).mean())
    support = float(action["support_score"])
    status = "PREDICTED" if support >= 0.15 and np.all(high < 0) else "ABSTAIN_UNCERTAIN_EFFECT"

    return {
        **baseline,
        "candidate_temperature_c": candidate.tolist(),
        "predicted_delta_c": predicted_delta,
        "interval_low_c": interval_low,
        "interval_high_c": interval_high,
        "support_score": support,
        "status": status,
        "action_kind": kind,
        "coverage_fraction": float(coverage_fraction),
        "selected_tile_ids": requested,
        "candra_effect_sha256": action_sha,
        "candra_source": action["source"],
    }
