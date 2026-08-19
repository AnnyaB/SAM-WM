from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .heatmap_view import validated_heatmap_feature_collection


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    timestamp: str
    activity_id: str
    content_sha256: str
    request_sha256: str
    map_data: dict[str, Any]


def _timestamp_from_request(request: dict[str, Any]) -> datetime:
    dt = request.get("date_time", {})
    if int(dt.get("filter_type", 0)) != 1:
        raise ValueError("timeline currently requires single-time heatmaps (filter_type=1)")
    return datetime.fromisoformat(f"{dt['start_date']}T{dt['start_time']}:00")


def _tile_signature(map_data: dict[str, Any]) -> tuple[str, ...]:
    features = map_data.get("features", [])
    ids = tuple(
        sorted(str(f.get("properties", {}).get("tile_id", f.get("id", ""))) for f in features)
    )
    if not ids or any(not x for x in ids):
        raise ValueError("heatmap frame has missing tile IDs")
    return ids


def load_recorded_heatmap_timeline(
    evidence_root: str | Path,
    *,
    limit: int = 48,
) -> list[ThermalFrame]:
    """Load the most recent fixed-grid sequence of real recorded TCM heatmaps.

    Frames from different tile grids are never mixed. The function chooses the
    tile signature whose latest valid frame is most recent, then returns its
    chronological tail.
    """
    root = Path(evidence_root)
    folder = root / "fortyguard_live"
    candidates: list[tuple[datetime, tuple[str, ...], ThermalFrame]] = []
    for meta_path in sorted(folder.glob("*.provenance.json")):
        try:
            prov = json.loads(meta_path.read_text(encoding="utf-8"))
            content_hash = str(prov["content_sha256"])
            request_hash = str(prov["request_sha256"])
            activity_id = str(prov["activity_id"])
            response_path = folder / f"{content_hash}.json"
            request_path = root / "requests" / f"{request_hash}.json"
            if not response_path.exists() or not request_path.exists():
                continue
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if request.get("analytic_type", "tcm") != "tcm":
                continue
            timestamp = _timestamp_from_request(request)
            response = json.loads(response_path.read_text(encoding="utf-8"))
            map_data = validated_heatmap_feature_collection(response["data"]["result"]["map_data"])
            signature = _tile_signature(map_data)
            frame = ThermalFrame(
                timestamp=timestamp.isoformat(),
                activity_id=activity_id,
                content_sha256=content_hash,
                request_sha256=request_hash,
                map_data=map_data,
            )
            candidates.append((timestamp, signature, frame))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Invalid/irrelevant evidence is excluded; nothing synthetic replaces it.
            continue
    if not candidates:
        return []
    latest_by_signature: dict[tuple[str, ...], datetime] = {}
    for ts, sig, _ in candidates:
        latest_by_signature[sig] = max(ts, latest_by_signature.get(sig, ts))
    chosen = max(latest_by_signature, key=latest_by_signature.get)  # type: ignore[arg-type]
    frames = [frame for ts, sig, frame in sorted(candidates, key=lambda x: x[0]) if sig == chosen]
    return frames[-max(1, limit) :]
