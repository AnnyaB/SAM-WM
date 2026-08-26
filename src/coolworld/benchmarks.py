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
    timestamps: np.ndarray  # datetime64[ns], [T]
    temperature: np.ndarray  # float32 [T,N]
    rh: np.ndarray  # float32 [T,N], nan if unavailable
    observed_mask: np.ndarray  # bool [T,N]
    station_ids: tuple[str, ...]
    lat: np.ndarray  # float32 [N]
    lon: np.ndarray  # float32 [N]
    elevation: np.ndarray  # float32 [N], nan allowed
    edge_index: torch.Tensor  # [2,E]
    edge_attr: torch.Tensor  # [E,2]


def md5(path: Path) -> str:
    h = hashlib.md5()
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
        raise FileNotFoundError(
            "Freiburg files missing; run with internet or place the exact Zenodo files in data/freiburg"
        )

    df = pd.read_csv(data_path)
    required = {"datetime_UTC", "station_id", "variable", "value", "data_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Freiburg schema mismatch; missing {sorted(required - set(df.columns))}")
    df["datetime_UTC"] = pd.to_datetime(df["datetime_UTC"], utc=True).dt.tz_localize(None)
    # WSN stations are FR****; exclude DWD numeric stations and any non-WSN rows.
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
        return val.to_numpy(np.float32), (typ.to_numpy(object) == "observed") & np.isfinite(
            val.to_numpy(float)
        )

    ta, observed = pivot("Ta_degC")
    rh, _ = pivot("RH_percent")
    if np.isnan(ta).mean() > 0.02:
        raise ValueError(
            "Freiburg temperature has unexpected residual missingness after curated gap-filling"
        )

    stats = pd.read_csv(stats_path)
    if "station_id" not in stats.columns:
        raise ValueError("Freiburg statistics file missing station_id")
    stats = stats.set_index("station_id").reindex(ids)
    lat = stats["latitude_degN"].to_numpy(np.float32)
    lon = stats["longitude_degE"].to_numpy(np.float32)
    elev = stats["elevation_masl"].to_numpy(np.float32)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Freiburg coordinates missing")
    edge_index, edge_attr = knn_graph(lat, lon, k)
    return UrbanDataset(
        "freiburg", idx.to_numpy(), ta, rh, observed, ids, lat, lon, elev, edge_index, edge_attr
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
        # Identify the temperature CSV by width and parse date/time.
        data_df = None
        for n in csvs:
            raw = zf.read(n)
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
        meta_raw = zf.read(excels[0])
        meta = pd.read_excel(io.BytesIO(meta_raw))

    cols = list(data_df.columns)
    date_col, time_col = cols[0], cols[1]
    ts = pd.to_datetime(
        data_df[date_col].astype(str) + " " + data_df[time_col].astype(str),
        dayfirst=True,
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)
    if ts.isna().mean() > 0.01:
        raise ValueError("Novi Sad timestamp parse failed")
    sensor_cols = cols[2:]
    ta = data_df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    observed = np.isfinite(ta)
    # Published hourly file is cleaned/gap-filled. It is not raw sensor truth; retain this fact in reporting.
    ta = pd.DataFrame(ta).interpolate(limit_direction="both").to_numpy(np.float32)

    # Robustly discover ID/lat/lon columns from metadata.
    mcols = {str(c).lower(): c for c in meta.columns}

    def find_col(tokens: tuple[str, ...]) -> str | None:
        for low, orig in mcols.items():
            if all(t in low for t in tokens):
                return orig
        return None

    idc = find_col(("id",)) or meta.columns[0]
    latc = find_col(("lat",))
    lonc = find_col(("lon",))
    if latc is None or lonc is None:
        raise ValueError(f"Novi Sad metadata missing coordinate columns: {list(meta.columns)}")
    meta_ids = meta[idc].astype(str).str.strip()
    lookup = {x: i for i, x in enumerate(meta_ids)}
    order = []
    for c in sensor_cols:
        key = str(c).strip()
        if key not in lookup:
            # Normalize punctuation/numeric Excel artefacts.
            hit = [x for x in lookup if re.sub(r"\W", "", x) == re.sub(r"\W", "", key)]
            if not hit:
                raise ValueError(f"Novi Sad sensor {key!r} missing from metadata")
            key = hit[0]
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
    rh = np.full_like(ta, np.nan, dtype=np.float32)
    return UrbanDataset(
        "novisad",
        ts.to_numpy(),
        ta,
        rh,
        observed,
        tuple(map(str, sensor_cols)),
        lat,
        lon,
        elev,
        edge_index,
        edge_attr,
    )


def _parse_sef_file(path: Path) -> tuple[str, float, float, pd.Series]:
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
        # SEF v1 normally has 12 header lines.
        data_start = 12
    station = headers.get("id", path.stem)
    lat = float(headers.get("lat", headers.get("latitude", "nan")))
    lon = float(headers.get("lon", headers.get("longitude", "nan")))
    table = pd.read_csv(io.StringIO("\n".join(lines[data_start:])), sep="\t")
    lower = {str(c).lower(): c for c in table.columns}
    needed = ["year", "month", "day", "hour", "minute", "value"]
    if not all(k in lower for k in needed):
        raise ValueError(f"SEF columns missing in {path.name}: {list(table.columns)}")
    dt = pd.to_datetime(
        dict(
            year=table[lower["year"]],
            month=table[lower["month"]],
            day=table[lower["day"]],
            hour=table[lower["hour"]],
            minute=table[lower["minute"]],
        ),
        utc=True,
        errors="coerce",
    ).dt.tz_localize(None)
    values = pd.to_numeric(table[lower["value"]], errors="coerce")
    s = pd.Series(values.to_numpy(float), index=dt).dropna().sort_index()
    return station, lat, lon, s


def load_fairurbtemp(root: str | Path, *, city: str | None = None, k: int = 4) -> UrbanDataset:
    """Load one extracted FAIRUrbTemp city using its hourly SEF files.

    The BORIS portal distributes 12 city archives. This loader intentionally does not guess
    a city archive URL: download/extract the official DOI 10.48620/93247 into `root`, then
    point this function at that extracted tree. It discovers hourly SEF temperature files,
    applies no fine-tuning, and fails closed on ambiguous layouts.
    """
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
    for p in candidates:
        try:
            station, lat, lon, s = _parse_sef_file(p)
        except Exception:
            continue
        if np.isfinite(lat) and np.isfinite(lon) and len(s) >= 24 * 14:
            parsed.append((station, lat, lon, s))
    if len(parsed) < 5:
        raise ValueError(
            "FAIRUrbTemp adapter could not identify >=5 hourly temperature SEF station files; inspect official extracted archive layout"
        )
    start = max(x[3].index.min() for x in parsed)
    end = min(x[3].index.max() for x in parsed)
    idx = pd.date_range(start.ceil("h"), end.floor("h"), freq="h")
    if len(idx) < 24 * 14:
        raise ValueError("FAIRUrbTemp stations lack a common >=14-day hourly interval")
    station_ids, lat, lon, cols = [], [], [], []
    for sid, la, lo, s in parsed:
        rs = s.resample("h").mean().reindex(idx)
        if rs.notna().mean() >= 0.8:
            station_ids.append(sid)
            lat.append(la)
            lon.append(lo)
            cols.append(rs.to_numpy(np.float32))
    if len(cols) < 5:
        raise ValueError("FAIRUrbTemp common interval has too few sufficiently complete stations")
    ta = np.stack(cols, axis=1)
    observed = np.isfinite(ta)
    # Context may be interpolated for continuity, but observed_mask keeps evaluation honest.
    ta_fill = (
        pd.DataFrame(ta)
        .interpolate(limit=3, limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )
    lat_a, lon_a = np.asarray(lat, np.float32), np.asarray(lon, np.float32)
    edge_index, edge_attr = knn_graph(lat_a, lon_a, k)
    rh = np.full_like(ta_fill, np.nan, np.float32)
    return UrbanDataset(
        "fairurbtemp",
        idx.to_numpy(),
        ta_fill,
        rh,
        observed,
        tuple(station_ids),
        lat_a,
        lon_a,
        np.full(len(cols), np.nan, np.float32),
        edge_index,
        edge_attr,
    )


def save_manifest(ds: UrbanDataset, path: str | Path) -> None:
    payload = {
        "name": ds.name,
        "n_timestamps": int(len(ds.timestamps)),
        "n_stations": int(len(ds.station_ids)),
        "timestamp_start": str(ds.timestamps[0]),
        "timestamp_end": str(ds.timestamps[-1]),
        "station_ids": list(ds.station_ids),
        "observed_target_fraction": float(ds.observed_mask.mean()),
        "edge_count": int(ds.edge_index.shape[1]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
