from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coolworld.evidence import sha256_file  # noqa: E402
from coolworld.product_api import product_state  # noqa: E402
from coolworld.provider import recorded_heatmap_frames  # noqa: E402

EXPECTED_MODEL = "SAM-WM"
EXPECTED_SEED = 42
EXPECTED_CHECKPOINT_SHA256 = "2be783f8a3b7f755a72a98949397c67dfec3a66a6400d8b98e1e732e0d8b708f"
MIN_COMPATIBLE_FRAMES = 65

CHECKPOINT = ROOT / "artifacts/deployment/best.pt"
PROMOTION = ROOT / "artifacts/deployment/PROMOTION_MANIFEST.json"
REPLAY = ROOT / "artifacts/deployment/fortyguard_replay.json"
EVIDENCE = ROOT / "artifacts/fortyguard"


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid runtime JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"runtime JSON is not an object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_checkpoint() -> str:
    require(CHECKPOINT.is_file(), f"missing checkpoint: {CHECKPOINT}")
    digest = sha256_file(CHECKPOINT)
    require(
        digest == EXPECTED_CHECKPOINT_SHA256,
        f"checkpoint SHA mismatch: {digest}",
    )
    return digest


def verify_promotion(checkpoint_sha: str) -> dict[str, Any]:
    payload = load_json(PROMOTION)
    require(
        payload.get("protocol") == "SAM_WM_DEPLOYMENT_PROMOTION_V1",
        "unexpected promotion protocol",
    )
    require(payload.get("model") == EXPECTED_MODEL, "promotion model mismatch")
    require(payload.get("selected_seed") == EXPECTED_SEED, "selected seed mismatch")
    require(
        payload.get("checkpoint_sha256") == checkpoint_sha,
        "promotion/checkpoint SHA mismatch",
    )
    return payload


def verify_timeline() -> tuple[list[dict[str, Any]], str]:
    frames = recorded_heatmap_frames(EVIDENCE, limit=1000)
    require(
        len(frames) >= MIN_COMPATIBLE_FRAMES,
        f"need >= {MIN_COMPATIBLE_FRAMES} compatible real frames; found {len(frames)}",
    )

    selected = frames[-MIN_COMPATIBLE_FRAMES:]
    try:
        stamps = [datetime.fromisoformat(str(frame["timestamp"])) for frame in selected]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("provider timeline contains invalid timestamps") from exc

    require(
        all(
            (right - left).total_seconds() == 3600
            for left, right in zip(stamps, stamps[1:], strict=False)
        ),
        "provider timeline is not consecutive hourly",
    )

    signatures = {str(frame.get("grid_signature")) for frame in selected}
    require(len(signatures) == 1, "provider timeline grid changed")
    signature = next(iter(signatures))
    require(signature and signature != "None", "provider grid signature missing")

    tile_counts = {len((frame.get("map_data") or {}).get("features") or []) for frame in selected}
    require(len(tile_counts) == 1, "provider tile count changed")
    require(next(iter(tile_counts)) > 1, "provider grid has too few tiles")

    return selected, signature


def verify_replay(checkpoint_sha: str, grid_signature: str) -> dict[str, Any]:
    payload = load_json(REPLAY)
    require(
        payload.get("protocol") == "SAM_WM_FORTYGUARD_REPLAY_V2",
        "unexpected provider replay protocol",
    )
    require(payload.get("model") == EXPECTED_MODEL, "provider replay model mismatch")
    require(
        payload.get("checkpoint_sha256") == checkpoint_sha,
        "provider replay/checkpoint SHA mismatch",
    )
    require(
        payload.get("grid_signature") == grid_signature,
        "provider replay/grid signature mismatch",
    )

    coverage = float(payload["conformal_coverage"])
    minimum_coverage = float(payload["minimum_required_coverage"])
    ratio = float(payload["mae_to_radius_ratio"])
    maximum_ratio = float(payload["maximum_allowed_mae_to_radius_ratio"])

    recomputed_pass = coverage >= minimum_coverage and ratio <= maximum_ratio
    expected_status = "PASS" if recomputed_pass else "FAIL"
    require(
        payload.get("status") == expected_status,
        "provider replay status disagrees with its frozen gate definition",
    )
    return payload


def main() -> None:
    checkpoint_sha = verify_checkpoint()
    promotion = verify_promotion(checkpoint_sha)
    frames, signature = verify_timeline()
    replay = verify_replay(checkpoint_sha, signature)
    state = product_state()

    require(state["model_bundle_promoted"] is True, "promoted model bundle not ready")
    require(state["research_forecast_ready"] is True, "research forecast should be ready")
    require(
        state["operational_certified"] is (replay["status"] == "PASS"),
        "product operational state disagrees with replay artifact",
    )

    report = {
        "status": "PASS",
        "model": EXPECTED_MODEL,
        "selected_seed": promotion["selected_seed"],
        "checkpoint_sha256": checkpoint_sha,
        "compatible_real_frames": len(recorded_heatmap_frames(EVIDENCE, limit=1000)),
        "verified_consecutive_frames": len(frames),
        "grid_signature": signature,
        "tile_count": len(frames[-1]["map_data"]["features"]),
        "research_forecast_ready": state["research_forecast_ready"],
        "operational_certified": state["operational_certified"],
        "provider_replay_status": replay["status"],
        "provider_replay_coverage": replay["conformal_coverage"],
        "provider_replay_minimum_coverage": replay["minimum_required_coverage"],
        "provider_replay_mae_to_radius_ratio": replay["mae_to_radius_ratio"],
        "provider_replay_maximum_ratio": replay["maximum_allowed_mae_to_radius_ratio"],
        "causal_action_ready": state["causal_action_ready"],
        "live_provider_api_enabled": state["live_provider_api_enabled"],
        "network_calls_made": 0,
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
