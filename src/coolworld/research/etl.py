from __future__ import annotations

import json
import math
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coolworld.grid import grid_signature_from_centroids
from coolworld.ml.schema import FeatureSchema


def _centroid_from_polygon(coords: list[Any]) -> tuple[float, float]:
    ring = coords[0]
    xs = [float(p[0]) for p in ring[:-1] or ring]
    ys = [float(p[1]) for p in ring[:-1] or ring]
    return float(np.mean(xs)), float(np.mean(ys))


def heatmap_evidence_to_table(evidence_root: str | Path) -> pd.DataFrame:
    """Turn recorded real FortyGuard TCM responses into a time/tile feature table."""
    root = Path(evidence_root)
    rows: list[dict[str, Any]] = []
    for meta in sorted((root / "fortyguard_live").glob("*.provenance.json")):
        prov = json.loads(meta.read_text(encoding="utf-8"))
        response_path = root / "fortyguard_live" / f"{prov['content_sha256']}.json"
        request_path = root / "requests" / f"{prov['request_sha256']}.json"
        if not response_path.exists() or not request_path.exists():
            continue
        response = json.loads(response_path.read_text(encoding="utf-8"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("analytic_type", "tcm") != "tcm":
            continue
        dt = request["date_time"]
        if int(dt.get("filter_type", 0)) != 1:
            continue
        timestamp = datetime.fromisoformat(f"{dt['start_date']}T{dt['start_time']}:00")
        result = response.get("data", {}).get("result", {})
        features = result.get("map_data", {}).get("features", [])
        for f in features:
            props = f.get("properties", {})
            temp = props.get("average_temperature")
            geom = f.get("geometry", {})
            if not isinstance(temp, (int, float)) or geom.get("type") != "Polygon":
                continue
            lon, lat = _centroid_from_polygon(geom["coordinates"])
            hour = timestamp.hour + timestamp.minute / 60.0
            rows.append(
                {
                    "timestamp": timestamp,
                    "tile_id": str(props.get("tile_id", f.get("id", ""))),
                    "temperature_c": float(temp),
                    "time_sin": math.sin(2 * math.pi * hour / 24.0),
                    "time_cos": math.cos(2 * math.pi * hour / 24.0),
                    "lat_scaled": lat / 90.0,
                    "lon_scaled": lon / 180.0,
                    "activity_id": prov["activity_id"],
                    "content_sha256": prov["content_sha256"],
                    "request_sha256": prov["request_sha256"],
                }
            )
    if not rows:
        raise ValueError("no usable real single-hour TCM heatmap evidence found")
    table = pd.DataFrame(rows).sort_values(["timestamp", "tile_id"]).reset_index(drop=True)
    return table


def build_sequence_bundle(
    table: pd.DataFrame,
    output_npz: str | Path,
    output_manifest: str | Path,
    *,
    sequence_len: int,
    intervention_log: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a fixed-AOI real sequence bundle for world-model training.

    If no real intervention log is supplied, action features are all zero. That
    is valid observational training, but it provides no evidence for non-zero
    interventions; downstream support checks should therefore reject such
    counterfactual actions.
    """
    required = {
        "timestamp",
        "tile_id",
        "temperature_c",
        "time_sin",
        "time_cos",
        "lat_scaled",
        "lon_scaled",
        "content_sha256",
    }
    if not required.issubset(table.columns):
        raise ValueError(f"table missing columns: {sorted(required.difference(table.columns))}")
    times = sorted(pd.to_datetime(table["timestamp"]).unique())
    if len(times) < sequence_len:
        raise ValueError("not enough timestamps for one sequence")
    tile_sets = [
        set(table[pd.to_datetime(table["timestamp"]) == t]["tile_id"].astype(str)) for t in times
    ]
    common = set.intersection(*tile_sets)
    if not common:
        raise ValueError("no common tiles across timestamps")
    tile_ids = sorted(common)
    n = len(tile_ids)
    id_to_idx = {x: i for i, x in enumerate(tile_ids)}

    action_cols = ["shade", "tree_canopy", "reflective_pavement"]
    intervention_lookup: dict[tuple[pd.Timestamp, str], np.ndarray] = {}
    if intervention_log is not None:
        need = {"timestamp", "tile_id", *action_cols}
        if not need.issubset(intervention_log.columns):
            missing = sorted(need.difference(intervention_log.columns))
            raise ValueError(f"intervention log missing columns: {missing}")
        for row in intervention_log.itertuples(index=False):
            key = (pd.Timestamp(row.timestamp), str(row.tile_id))
            vec = np.asarray([getattr(row, c) for c in action_cols], dtype=np.float32)
            if np.any((vec < 0) | (vec > 1)):
                raise ValueError("intervention action values must be in [0,1]")
            intervention_lookup[key] = vec

    steps = []
    source_hashes: set[str] = set()
    static = np.zeros((n, 2), dtype=np.float32)
    for t in times:
        frame = table[pd.to_datetime(table["timestamp"]) == t]
        dynamic = np.zeros((n, 3), dtype=np.float32)
        actions = np.zeros((n, 3), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for row in frame.itertuples(index=False):
            tile_id = str(row.tile_id)
            if tile_id not in id_to_idx:
                continue
            i = id_to_idx[tile_id]
            dynamic[i] = [float(row.temperature_c), float(row.time_sin), float(row.time_cos)]
            static[i] = [float(row.lat_scaled), float(row.lon_scaled)]
            mask[i] = True
            source_hashes.add(str(row.content_sha256))
            actions[i] = intervention_lookup.get(
                (pd.Timestamp(t), tile_id), np.zeros(3, dtype=np.float32)
            )
        steps.append((dynamic, actions, mask))

    samples = len(times) - sequence_len + 1
    dyn = np.stack(
        [[steps[j][0] for j in range(i, i + sequence_len)] for i in range(samples)]
    ).astype(np.float32)
    act = np.stack(
        [[steps[j][1] for j in range(i, i + sequence_len)] for i in range(samples)]
    ).astype(np.float32)
    msk = np.stack(
        [[steps[j][2] for j in range(i, i + sequence_len)] for i in range(samples)]
    ).astype(bool)
    sta = np.repeat(static[None], samples, axis=0).astype(np.float32)

    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dynamic=dyn, static=sta, actions=act, mask=msk)
    file_hash = sha256(out.read_bytes()).hexdigest()
    schema = FeatureSchema(
        ("temperature_c", "time_sin", "time_cos"), ("lat_scaled", "lon_scaled"), tuple(action_cols)
    )
    diffs_minutes = [
        float((pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 60.0)
        for i in range(1, len(times))
    ]
    cadence_minutes_median = float(np.median(diffs_minutes)) if diffs_minutes else 0.0
    if cadence_minutes_median <= 0:
        raise ValueError("timestamps must have a positive cadence for future rollout")
    centroid_rows = [
        (tile_id, float(static[i, 1] * 180.0), float(static[i, 0] * 90.0))
        for i, tile_id in enumerate(tile_ids)
    ]
    grid_signature = grid_signature_from_centroids(centroid_rows)
    manifest = {
        "dataset_id": f"urban-thermal-{file_hash[:12]}",
        "file_sha256": file_hash,
        "source_records": [
            {
                "kind": "fortyguard_recorded_live",
                "content_sha256": h,
                "source_reference": "local immutable FortyGuard evidence store",
            }
            for h in sorted(source_hashes)
        ],
        "schema": schema.to_dict(),
        "sequence_len": sequence_len,
        "tiles": n,
        "tile_ids": tile_ids,
        "grid_signature": grid_signature,
        "timestamps": len(times),
        "timestamp_start": str(pd.Timestamp(times[0]).isoformat()),
        "timestamp_end": str(pd.Timestamp(times[-1]).isoformat()),
        "cadence_minutes_median": cadence_minutes_median,
        "sequence_stride": 1,
        "nonzero_action_records": int(np.count_nonzero(act)),
    }
    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
