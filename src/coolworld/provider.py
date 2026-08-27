from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .deployment import (
    DeploymentError,
    baseline_forecast,
    canonical_map_data,
    grid_signature,
    observed_temperature,
    stable_tile_id,
    validate_deployment_bundle,
)
from .evidence import sha256_file
from .experiment import load_checkpoint
from .fortyguard import completed_heatmap_records

MIN_REPLAY_WINDOWS = 12
MIN_REPLAY_COVERAGE = 0.80


def _request_timestamp(request: dict[str, Any]) -> str | None:
    date_time = request.get("date_time", {})
    date = date_time.get("start_date")
    clock = date_time.get("start_time")
    if not date or not clock:
        return None
    value = f"{date}T{clock}"
    try:
        return np.datetime_as_string(np.datetime64(value, "s"), unit="s")
    except ValueError:
        return None


def recorded_heatmap_frames(evidence_dir: Path, limit: int = 240) -> list[dict[str, Any]]:
    """Return the latest same-grid immutable FortyGuard temperature evidence timeline."""
    frames: list[dict[str, Any]] = []
    for record in completed_heatmap_records(evidence_dir):
        timestamp = _request_timestamp(record.get("request_payload", {}))
        result = record.get("response", {}).get("data", {}).get("result", {})
        map_data = result.get("map_data") if isinstance(result, dict) else None
        if timestamp is None or not isinstance(map_data, dict):
            continue
        try:
            canonical = canonical_map_data(map_data)
            signature = grid_signature(canonical)
        except DeploymentError:
            continue
        frames.append(
            {
                "timestamp": timestamp,
                "activity_id": record.get("activity_id"),
                "request_sha256": record.get("request_sha256"),
                "content_sha256": record.get("content_sha256"),
                "grid_signature": signature,
                "map_data": canonical,
            }
        )

    frames.sort(key=lambda frame: frame["timestamp"])
    if not frames:
        return []
    signature = frames[-1]["grid_signature"]
    compatible = [frame for frame in frames if frame["grid_signature"] == signature]
    return compatible[-max(1, min(int(limit), 1000)) :]


def _frame_temperatures(frame: dict[str, Any], tile_ids: list[str]) -> np.ndarray:
    canonical = canonical_map_data(frame["map_data"])
    by_id = {stable_tile_id(feature): feature for feature in canonical["features"]}
    if set(by_id) != set(tile_ids):
        raise DeploymentError("PROVIDER_REPLAY_GRID_CHANGED")
    return np.asarray([observed_temperature(by_id[tile_id]) for tile_id in tile_ids], dtype=float)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def evaluate_provider_replay(
    checkpoint: Path,
    calibration: Path,
    evaluation: Path,
    frames: list[dict[str, Any]],
    out: Path,
) -> dict[str, Any]:
    """Validate transfer to real FortyGuard TCM before enabling live predicted mode.

    This is an application-domain compatibility gate, not a third paper OOD benchmark and
    not evidence of a causal cooling effect.
    """
    bundle = validate_deployment_bundle(checkpoint, calibration, evaluation)
    _, _, cfg, _ = load_checkpoint(checkpoint)
    context = int(cfg["context_hours"])
    horizon = int(cfg["horizon_hours"])
    if len(frames) < context + horizon + MIN_REPLAY_WINDOWS - 1:
        raise DeploymentError(
            f"PROVIDER_REPLAY_REQUIRES_AT_LEAST_{context + horizon + MIN_REPLAY_WINDOWS - 1}_FRAMES"
        )

    timestamps = np.asarray([np.datetime64(frame["timestamp"], "s") for frame in frames])
    if np.isnat(timestamps).any() or not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise DeploymentError("PROVIDER_REPLAY_REQUIRES_CONSECUTIVE_HOURLY_FRAMES")
    signatures = {
        frame.get("grid_signature") or grid_signature(frame["map_data"]) for frame in frames
    }
    if len(signatures) != 1:
        raise DeploymentError("PROVIDER_REPLAY_GRID_CHANGED")

    model_errors: list[np.ndarray] = []
    persistence_errors: list[np.ndarray] = []
    covered = 0
    total = 0
    windows = 0
    first = max(context, len(frames) - (MIN_REPLAY_WINDOWS + horizon - 1))
    for start in range(first, len(frames) - horizon + 1):
        context_frames = frames[start - context : start]
        prediction = baseline_forecast(checkpoint, calibration, evaluation, context_frames)
        tile_ids = list(prediction["tile_ids"])
        pred = np.asarray(prediction["baseline_temperature_c"], dtype=float)
        actual = np.stack(
            [_frame_temperatures(frames[start + step], tile_ids) for step in range(horizon)],
            axis=0,
        )
        expected = [
            np.datetime_as_string(timestamps[start + step], unit="s") for step in range(horizon)
        ]
        if expected != list(prediction["future_timestamps"]):
            raise DeploymentError("PROVIDER_REPLAY_TIMESTAMP_MISMATCH")
        last = _frame_temperatures(frames[start - 1], tile_ids)
        persistence = np.repeat(last[None, :], horizon, axis=0)
        error = pred - actual
        model_errors.append(error.reshape(-1))
        persistence_errors.append((persistence - actual).reshape(-1))
        radius = float(prediction["baseline_conformal_radius_c"])
        covered += int((np.abs(error) <= radius).sum())
        total += int(error.size)
        windows += 1

    if windows < MIN_REPLAY_WINDOWS:
        raise DeploymentError("PROVIDER_REPLAY_WINDOW_COUNT_INSUFFICIENT")
    model_error = np.concatenate(model_errors)
    persistence_error = np.concatenate(persistence_errors)
    model_mae = float(np.abs(model_error).mean())
    persistence_mae = float(np.abs(persistence_error).mean())
    coverage = covered / total
    pass_gate = model_mae <= persistence_mae and coverage >= MIN_REPLAY_COVERAGE

    payload = {
        "protocol": "SAM_WM_FORTYGUARD_REPLAY_V1",
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "grid_signature": next(iter(signatures)),
        "frame_count": len(frames),
        "window_count": windows,
        "context_hours": context,
        "horizon_hours": horizon,
        "model_mae_c": model_mae,
        "model_rmse_c": float(np.sqrt(np.square(model_error).mean())),
        "model_bias_c": float(model_error.mean()),
        "persistence_mae_c": persistence_mae,
        "conformal_coverage": coverage,
        "minimum_required_coverage": MIN_REPLAY_COVERAGE,
        "status": "PASS" if pass_gate else "FAIL",
        "claim_boundary": (
            "Provider replay checks operational transfer to recorded FortyGuard TCM fields only. "
            "It is not causal intervention evidence and is not a planetary-scale validation."
        ),
    }
    _write_json(out, payload)
    return payload


def validate_provider_replay(path: Path, checkpoint_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise DeploymentError("FORTYGUARD_REPLAY_REQUIRED")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError("FORTYGUARD_REPLAY_INVALID") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != "SAM_WM_FORTYGUARD_REPLAY_V1":
        raise DeploymentError("FORTYGUARD_REPLAY_INVALID")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise DeploymentError("FORTYGUARD_REPLAY_CHECKPOINT_HASH_MISMATCH")
    if payload.get("status") != "PASS":
        raise DeploymentError("FORTYGUARD_REPLAY_GATE_FAILED")
    if int(payload.get("window_count", 0)) < MIN_REPLAY_WINDOWS:
        raise DeploymentError("FORTYGUARD_REPLAY_WINDOW_COUNT_INSUFFICIENT")
    coverage = float(payload.get("conformal_coverage", float("nan")))
    if not np.isfinite(coverage) or coverage < MIN_REPLAY_COVERAGE:
        raise DeploymentError("FORTYGUARD_REPLAY_COVERAGE_INSUFFICIENT")
    return {**payload, "artifact_sha256": sha256_file(path)}
