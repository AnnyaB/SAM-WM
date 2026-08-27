from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import knn_graph


@dataclass(frozen=True)
class UrbanDataset:
    name: str
    timestamps: np.ndarray
    temperature: np.ndarray
    rh: np.ndarray
    observed_mask: np.ndarray
    station_ids: tuple[str, ...]
    lat: np.ndarray
    lon: np.ndarray
    elevation: np.ndarray
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    source: str = ""


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, expected_md5: str | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (expected_md5 is None or md5(dst) == expected_md5):
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if expected_md5 and md5(tmp) != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dst.name}")
    tmp.replace(dst)
    return dst


def _complete_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", inclusive="both", tz="UTC").tz_localize(None)


def _canonicalize_freiburg_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official Freiburg CSV into the published long-form contract.

    Zenodo documents the semantic fields ``datetime_UTC``, ``station_id``,
    ``variable``, ``value`` and ``data_type``. CSV readers should not make the
    benchmark depend on superficial header case/BOM/whitespace, and some exported
    copies expose the two variables as wide ``Ta_degC``/``RH_percent`` columns.
    Both representations carry the same official semantics and are normalized here.
    Unknown layouts fail closed with the observed columns instead of being guessed.
    """
    canonical_names = {
        "datetime_utc": "datetime_UTC",
        "station_id": "station_id",
        "variable": "variable",
        "value": "value",
        "data_type": "data_type",
        "ta_degc": "Ta_degC",
        "rh_percent": "RH_percent",
    }
    rename: dict[object, str] = {}
    for column in raw.columns:
        normalized = str(column).lstrip("\ufeff").strip().casefold()
        if normalized in canonical_names:
            rename[column] = canonical_names[normalized]

    df = raw.rename(columns=rename).copy()
    if df.columns.duplicated().any():
        duplicated = [str(column) for column in df.columns[df.columns.duplicated()].tolist()]
        raise ValueError(f"Freiburg schema has duplicate canonical columns: {duplicated}")

    common = {"datetime_UTC", "station_id", "data_type"}
    if not common.issubset(df.columns):
        missing = sorted(common - set(df.columns))
        raise ValueError(
            "Freiburg schema mismatch; "
            f"missing {missing}; observed columns={list(map(str, raw.columns))}"
        )

    if {"variable", "value"}.issubset(df.columns):
        long = df[["datetime_UTC", "station_id", "variable", "value", "data_type"]].copy()
    elif {"Ta_degC", "RH_percent"}.issubset(df.columns):
        pieces: list[pd.DataFrame] = []
        for variable in ("Ta_degC", "RH_percent"):
            piece = df[["datetime_UTC", "station_id", variable, "data_type"]].copy()
            piece = piece.rename(columns={variable: "value"})
            piece["variable"] = variable
            pieces.append(piece)
        long = pd.concat(pieces, ignore_index=True)
        long = long[["datetime_UTC", "station_id", "variable", "value", "data_type"]]
    else:
        raise ValueError(
            "Freiburg schema mismatch; expected long variable/value fields or official "
            f"wide Ta_degC/RH_percent fields; observed columns={list(map(str, raw.columns))}"
        )

    variable_map = {"ta_degc": "Ta_degC", "rh_percent": "RH_percent"}
    normalized_variable = long["variable"].astype(str).str.strip().str.casefold()
    unexpected = sorted(set(normalized_variable) - set(variable_map))
    if unexpected:
        raise ValueError(f"Freiburg contains unexpected variables: {unexpected}")
    long["variable"] = normalized_variable.map(variable_map)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["data_type"] = long["data_type"].astype(str).str.strip().str.casefold()
    unexpected_types = sorted(set(long["data_type"].dropna()) - {"observed", "imputed"})
    if unexpected_types:
        raise ValueError(f"Freiburg contains unexpected data_type values: {unexpected_types}")

    duplicates = long.duplicated(["datetime_UTC", "station_id", "variable"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Freiburg contains duplicate station/time/variable rows; "
            f"count={int(duplicates.sum())}"
        )
    return long


def load_freiburg(
    root: str | Path = "data/freiburg", *, k: int = 4, download_if_missing: bool = True
) -> UrbanDataset:
    root = Path(root)
    data_name = "Freiburg_AWS_20220901_20230831_gap_filled_data_ta_rh_Plein_et_al.csv"
    stats_name = "Freiburg_AWS_20220901_20230831_annual_statistics_per_station_Plein_et_al.csv"
    data_path = root / data_name
    stats_path = root / stats_name
    if download_if_missing:
        download(
            f"https://zenodo.org/records/12732565/files/{data_name}?download=1",
            data_path,
            "840a2f677d43b1298f50f40f0a250d98",
        )
        download(
            f"https://zenodo.org/records/12732565/files/{stats_name}?download=1",
            stats_path,
            "4a70262921bd9a90513fe6cf25527163",
        )
    if not data_path.exists() or not stats_path.exists():
        raise FileNotFoundError("official Freiburg files missing")

    df = _canonicalize_freiburg_table(pd.read_csv(data_path))
    df["datetime_UTC"] = pd.to_datetime(
        df["datetime_UTC"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    if df["datetime_UTC"].isna().any():
        raise ValueError("Freiburg contains unparseable datetime_UTC values")
    df = df[df["station_id"].astype(str).str.fullmatch(r"FR[A-Z0-9]{4}", na=False)].copy()
    ids = tuple(sorted(df["station_id"].astype(str).unique()))
    if len(ids) < 30:
        raise ValueError(f"expected dense Freiburg WSN, found only {len(ids)} stations")
    idx = _complete_hourly_index("2022-09-01 00:00", "2023-08-31 23:00")

    def pivot(var: str) -> tuple[np.ndarray, np.ndarray]:
        sub = df[df["variable"] == var]
        val = sub.pivot(index="datetime_UTC", columns="station_id", values="value").reindex(
            index=idx, columns=ids
        )
        typ = sub.pivot(index="datetime_UTC", columns="station_id", values="data_type").reindex(
            index=idx, columns=ids
        )
        values = val.to_numpy(np.float32)
        observed = (typ.to_numpy(object) == "observed") & np.isfinite(values)
        return values, observed

    ta, observed = pivot("Ta_degC")
    rh, _ = pivot("RH_percent")
    if np.isnan(ta).mean() > 0.02:
        raise ValueError("unexpected Freiburg residual missingness after curated gap filling")

    stats = pd.read_csv(stats_path)
    normalized_stats = {str(c).lstrip("\ufeff").strip().casefold(): c for c in stats.columns}
    required_stats = {
        "station_id": "station_id",
        "latitude_degn": "latitude_degN",
        "longitude_dege": "longitude_degE",
        "elevation_masl": "elevation_masl",
    }
    rename_stats = {
        normalized_stats[key]: canonical
        for key, canonical in required_stats.items()
        if key in normalized_stats
    }
    stats = stats.rename(columns=rename_stats)
    missing_stats = sorted(set(required_stats.values()) - set(stats.columns))
    if missing_stats:
        raise ValueError(
            "Freiburg statistics schema mismatch; "
            f"missing {missing_stats}; observed columns={list(map(str, stats.columns))}"
        )
    stats = stats.set_index("station_id").reindex(ids)
    lat = pd.to_numeric(stats["latitude_degN"], errors="coerce").to_numpy(np.float32)
    lon = pd.to_numeric(stats["longitude_degE"], errors="coerce").to_numpy(np.float32)
    elev = pd.to_numeric(stats["elevation_masl"], errors="coerce").to_numpy(np.float32)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Freiburg coordinates missing")
    edge_index, edge_attr = knn_graph(lat, lon, k)
    return UrbanDataset(
        "freiburg",
        idx.to_numpy(),
        ta,
        rh,
        observed,
        ids,
        lat,
        lon,
        elev,
        edge_index,
        edge_attr,
        source="doi:10.5281/zenodo.12732565",
    )


def load_novisad(
    root: str | Path = "data/novisad", *, k: int = 4, download_if_missing: bool = True
) -> UrbanDataset:
    root = Path(root)
    archive = root / "NSUNET_Ta dataset_and_site metadata.zip"
    if download_if_missing:
        download(
            "https://zenodo.org/records/7738094/files/NSUNET_Ta%20dataset_and_site%20metadata.zip?download=1",
            archive,
            "a9e3574d500b0a621a209cc41c1d6fb8",
        )
    if not archive.exists():
        raise FileNotFoundError("Novi Sad archive missing")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        excels = [n for n in names if n.lower().endswith((".xlsx", ".xls"))]
        if not csvs or not excels:
            raise ValueError(f"unexpected Novi Sad archive contents: {names}")
        data_df = None
        for name in csvs:
            raw = zf.read(name)
            for sep in [",", ";", "\t"]:
                try:
                    cand = pd.read_csv(io.BytesIO(raw), sep=sep)
                except Exception:
                    continue
                if cand.shape[1] >= 14:
                    data_df = cand
                    break
            if data_df is not None:
                break
        if data_df is None:
            raise ValueError("could not locate Novi Sad hourly temperature table")
        meta = pd.read_excel(io.BytesIO(zf.read(excels[0])))

    cols = list(data_df.columns)
    ts = pd.to_datetime(
        data_df[cols[0]].astype(str) + " " + data_df[cols[1]].astype(str),
        dayfirst=True,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)
    if ts.isna().mean() > 0.01:
        raise ValueError("Novi Sad timestamp parse failed")
    sensor_cols = cols[2:]
    raw_ta = data_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    observed = np.isfinite(raw_ta)
    ta = pd.DataFrame(raw_ta).interpolate(limit_direction="both").to_numpy(np.float32)

    lower_cols = {str(c).lower(): c for c in meta.columns}

    def find_col(tokens: tuple[str, ...]) -> str | None:
        for low, orig in lower_cols.items():
            if all(t in low for t in tokens):
                return orig
        return None

    idc = find_col(("id",)) or meta.columns[0]
    latc, lonc = find_col(("lat",)), find_col(("lon",))
    if latc is None or lonc is None:
        raise ValueError(f"Novi Sad metadata missing coordinate columns: {list(meta.columns)}")
    lookup = {x: i for i, x in enumerate(meta[idc].astype(str).str.strip())}
    order = []
    for col in sensor_cols:
        key = str(col).strip()
        if key not in lookup:
            hits = [x for x in lookup if re.sub(r"\W", "", x) == re.sub(r"\W", "", key)]
            if not hits:
                raise ValueError(f"Novi Sad sensor {key!r} missing from metadata")
            key = hits[0]
        order.append(lookup[key])
    m = meta.iloc[order]
    lat = pd.to_numeric(m[latc], errors="coerce").to_numpy(np.float32)
    lon = pd.to_numeric(m[lonc], errors="coerce").to_numpy(np.float32)
    elevc = find_col(("elev",)) or find_col(("alt",))
    elev = (
        np.full(len(sensor_cols), np.nan, np.float32)
        if elevc is None
        else pd.to_numeric(m[elevc], errors="coerce").to_numpy(np.float32)
    )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Novi Sad coordinates invalid")
    edge_index, edge_attr = knn_graph(lat, lon, k)
    return UrbanDataset(
        "novisad",
        ts.to_numpy(),
        ta,
        np.full_like(ta, np.nan, np.float32),
        observed,
        tuple(map(str, sensor_cols)),
        lat,
        lon,
        elev,
        edge_index,
        edge_attr,
        source="doi:10.5281/zenodo.7738094",
    )


def _parse_sef_file(path: Path) -> tuple[str, float, float, pd.Series]:
    """Parse one FAIRUrbTemp hourly SEF file and remove QC-flagged observations.

    FAIRUrbTemp retains suspicious values and records QC flags in the observation-level
    Meta field. Values carrying a ``qc = ...`` entry are treated as unavailable rather
    than silently scored as ground truth.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("SEF"):
        raise ValueError(f"not SEF: {path}")
    headers: dict[str, str] = {}
    data_start = None
    for i, line in enumerate(lines):
        parts = line.split("\t")
        if parts and parts[0].strip().lower() in {"year", "yyyy"}:
            data_start = i
            break
        if len(parts) >= 2:
            headers[parts[0].strip().lower()] = parts[1].strip()
    if data_start is None:
        data_start = 12
    station = headers.get("id", path.stem)
    lat = float(headers.get("lat", headers.get("latitude", "nan")))
    lon = float(headers.get("lon", headers.get("longitude", "nan")))
    table = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), sep="\t")
    lower = {str(c).strip().lower(): c for c in table.columns}
    needed = ["year", "month", "day", "hour", "minute", "value"]
    if not all(k in lower for k in needed):
        raise ValueError(f"SEF columns missing in {path.name}: {list(table.columns)}")
    dt = pd.to_datetime(
        {
            "year": table[lower["year"]],
            "month": table[lower["month"]],
            "day": table[lower["day"]],
            "hour": table[lower["hour"]],
            "minute": table[lower["minute"]],
        },
        utc=True,
        errors="coerce",
    ).dt.tz_localize(None)
    values = pd.to_numeric(table[lower["value"]], errors="coerce")
    if "meta" in lower:
        meta = table[lower["meta"]].fillna("").astype(str)
        flagged = meta.str.contains(r"(?:^|\|)\s*qc\s*=", case=False, regex=True)
        values = values.mask(flagged)
    series = pd.Series(values.to_numpy(float), index=dt).dropna().sort_index()
    return station, lat, lon, series


def load_fairurbtemp(root: str | Path, *, city: str | None = None, k: int = 4) -> UrbanDataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"FAIRUrbTemp root not found: {root}")
    files = list(root.rglob("*.tsv"))
    if city:
        files = [p for p in files if city.lower() in str(p).lower()]
    hourly = [
        p
        for p in files
        if "hour" in str(p).lower()
        and ("temp" in str(p).lower() or "_ta" in p.name.lower() or "qc" in str(p).lower())
    ]
    candidates = hourly or files
    parsed = []
    for path in candidates:
        try:
            station, lat, lon, series = _parse_sef_file(path)
        except Exception:
            continue
        if np.isfinite(lat) and np.isfinite(lon) and len(series) >= 24 * 14:
            parsed.append((station, lat, lon, series))
    if len(parsed) < 5:
        raise ValueError("could not identify >=5 valid hourly FAIRUrbTemp temperature stations")
    start = max(x[3].index.min() for x in parsed)
    end = min(x[3].index.max() for x in parsed)
    idx = pd.date_range(start.ceil("h"), end.floor("h"), freq="h")
    if len(idx) < 24 * 14:
        raise ValueError("FAIRUrbTemp stations lack a common >=14-day interval")
    station_ids: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    cols: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for sid, lat, lon, series in parsed:
        resampled = series.resample("h").mean().reindex(idx)
        if resampled.notna().mean() >= 0.8:
            station_ids.append(sid)
            lats.append(lat)
            lons.append(lon)
            values = resampled.to_numpy(np.float32)
            masks.append(np.isfinite(values))
            cols.append(values)
    if len(cols) < 5:
        raise ValueError("FAIRUrbTemp common interval has too few sufficiently complete stations")
    raw = np.stack(cols, axis=1)
    observed = np.stack(masks, axis=1)
    filled = (
        pd.DataFrame(raw)
        .interpolate(limit=3, limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )
    lat_a, lon_a = np.asarray(lats, np.float32), np.asarray(lons, np.float32)
    edge_index, edge_attr = knn_graph(lat_a, lon_a, k)
    return UrbanDataset(
        "fairurbtemp",
        idx.to_numpy(),
        filled,
        np.full_like(filled, np.nan, np.float32),
        observed,
        tuple(station_ids),
        lat_a,
        lon_a,
        np.full(len(station_ids), np.nan, np.float32),
        edge_index,
        edge_attr,
        source="doi:10.48620/93247",
    )


def save_manifest(ds: UrbanDataset, path: str | Path) -> None:
    payload = {
        "dataset": ds.name,
        "source": ds.source,
        "n_timestamps": int(len(ds.timestamps)),
        "n_nodes": int(len(ds.station_ids)),
        "start": str(ds.timestamps[0]),
        "end": str(ds.timestamps[-1]),
        "observed_target_fraction": float(ds.observed_mask.mean()),
        "edge_count": int(ds.edge_index.shape[1]),
        "edge_attr": ["unit_x", "unit_y", "log1p_distance_km"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
