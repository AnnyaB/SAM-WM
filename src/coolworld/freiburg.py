from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import UrbanDataset, download
from .graph import knn_graph

DATA_NAME = "Freiburg_AWS_20220901_20230831_gap_filled_data_ta_rh_Plein_et_al.csv"
STATS_NAME = "Freiburg_AWS_20220901_20230831_annual_statistics_per_station_Plein_et_al.csv"
DATA_MD5 = "840a2f677d43b1298f50f40f0a250d98"
STATS_MD5 = "4a70262921bd9a90513fe6cf25527163"
SOURCE = "doi:10.5281/zenodo.12732565"
EXPECTED_STATIONS = 41
EXPECTED_HOURS = 8760
OFFICIAL_COLUMNS = ("datetime_UTC", "station_id", "variable", "value", "data_type")
RELEASE_HEADER = ("datetime_UTC", "station_id", "variable,value", "data_type")
VARIABLE_ALIASES = {
    "Ta_degC": "Ta_degC",
    "Ta_deg_C": "Ta_degC",
    "RH_percent": "RH_percent",
}


def _clean_header(row: list[str]) -> tuple[str, ...]:
    return tuple(field.lstrip("\ufeff").strip() for field in row)


def _inspect_csv_shape(path: Path, sample_rows: int = 32) -> tuple[tuple[str, ...], set[int]]:
    """Read physical CSV fields before pandas can infer an accidental index."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = _clean_header(next(reader))
        except StopIteration as exc:
            raise ValueError("Freiburg data file is empty") from exc

        widths: set[int] = set()
        for _, row in zip(range(sample_rows), reader, strict=False):
            if row:
                widths.add(len(row))

    if not widths:
        raise ValueError("Freiburg data file contains a header but no data rows")
    return header, widths


def read_freiburg_table(path: str | Path) -> pd.DataFrame:
    """Read the checksum-verified Freiburg release into its five semantic fields.

    Zenodo documents five fields. The released CSV can expose a quoted
    ``variable,value`` header as one physical header field while its data rows still
    contain five comma-separated fields. Pandas then silently shifts the first row
    field into the index. We detect that physical layout before pandas parsing and
    supply the documented five-column schema explicitly. Unknown layouts fail closed.

    The Zenodo record documents ``Ta_degC`` for temperature, while the released file
    observed in the Kaggle execution can contain ``Ta_deg_C``. Both are canonicalized
    to the documented internal label ``Ta_degC`` before tensor construction.
    """
    path = Path(path)
    header, row_widths = _inspect_csv_shape(path)

    if header == OFFICIAL_COLUMNS and row_widths == {5}:
        table = pd.read_csv(path)
    elif header == RELEASE_HEADER and row_widths == {5}:
        table = pd.read_csv(
            path,
            header=None,
            skiprows=1,
            names=list(OFFICIAL_COLUMNS),
        )
    else:
        raise ValueError(
            "Freiburg physical CSV layout mismatch; "
            f"header={list(header)}, sampled_row_widths={sorted(row_widths)}"
        )

    if tuple(map(str, table.columns)) != OFFICIAL_COLUMNS:
        raise ValueError(f"Freiburg semantic columns mismatch: {list(table.columns)}")

    table = table.copy()
    table["station_id"] = table["station_id"].astype(str).str.strip()
    raw_variables = table["variable"].astype(str).str.strip()
    table["data_type"] = table["data_type"].astype(str).str.strip().str.casefold()
    table["value"] = pd.to_numeric(table["value"], errors="coerce")

    invalid_station = ~table["station_id"].str.fullmatch(r"FR[A-Z0-9]{4}", na=False)
    if invalid_station.any():
        examples = table.loc[invalid_station, "station_id"].head(3).tolist()
        raise ValueError(f"Freiburg contains invalid station IDs: {examples}")

    unexpected_variables = sorted(set(raw_variables.dropna()) - set(VARIABLE_ALIASES))
    if unexpected_variables:
        raise ValueError(f"Freiburg contains unexpected variables: {unexpected_variables}")
    table["variable"] = raw_variables.map(VARIABLE_ALIASES)

    unexpected_types = sorted(set(table["data_type"].dropna()) - {"observed", "imputed"})
    if unexpected_types:
        raise ValueError(f"Freiburg contains unexpected data_type values: {unexpected_types}")

    if table["value"].isna().any():
        raise ValueError("Freiburg gap-filled release contains non-numeric values")

    table["datetime_UTC"] = pd.to_datetime(
        table["datetime_UTC"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    if table["datetime_UTC"].isna().any():
        raise ValueError("Freiburg contains unparseable datetime_UTC values")

    duplicates = table.duplicated(["datetime_UTC", "station_id", "variable"], keep=False)
    if duplicates.any():
        raise ValueError(
            f"Freiburg contains duplicate station/time/variable rows; count={int(duplicates.sum())}"
        )

    return table


def _complete_hourly_index() -> pd.DatetimeIndex:
    return pd.date_range(
        "2022-09-01 00:00",
        "2023-08-31 23:00",
        freq="h",
        inclusive="both",
        tz="UTC",
    ).tz_localize(None)


def load_freiburg(
    root: str | Path = "data/freiburg",
    *,
    k: int = 4,
    download_if_missing: bool = True,
) -> UrbanDataset:
    """Load the exact checksum-pinned Freiburg benchmark used by SAM-WM."""
    root = Path(root)
    data_path = root / DATA_NAME
    stats_path = root / STATS_NAME

    if download_if_missing:
        download(
            f"https://zenodo.org/records/12732565/files/{DATA_NAME}?download=1",
            data_path,
            DATA_MD5,
        )
        download(
            f"https://zenodo.org/records/12732565/files/{STATS_NAME}?download=1",
            stats_path,
            STATS_MD5,
        )

    if not data_path.exists() or not stats_path.exists():
        raise FileNotFoundError("official Freiburg files missing")

    df = read_freiburg_table(data_path)
    station_ids = tuple(sorted(df["station_id"].unique()))
    if len(station_ids) != EXPECTED_STATIONS:
        raise ValueError(
            f"expected {EXPECTED_STATIONS} Freiburg stations, found {len(station_ids)}"
        )

    index = _complete_hourly_index()
    if len(index) != EXPECTED_HOURS:
        raise RuntimeError("internal Freiburg hourly index invariant failed")

    def pivot(variable: str) -> tuple[np.ndarray, np.ndarray]:
        subset = df[df["variable"] == variable]
        values_frame = subset.pivot(
            index="datetime_UTC", columns="station_id", values="value"
        ).reindex(index=index, columns=station_ids)
        type_frame = subset.pivot(
            index="datetime_UTC", columns="station_id", values="data_type"
        ).reindex(index=index, columns=station_ids)

        values = values_frame.to_numpy(np.float32)
        observed = (type_frame.to_numpy(object) == "observed") & np.isfinite(values)
        return values, observed

    temperature, observed = pivot("Ta_degC")
    rh, _ = pivot("RH_percent")

    expected_shape = (EXPECTED_HOURS, EXPECTED_STATIONS)
    if temperature.shape != expected_shape or rh.shape != expected_shape:
        raise ValueError(
            "Freiburg tensor shape mismatch; "
            f"temperature={temperature.shape}, rh={rh.shape}, expected={expected_shape}"
        )
    if np.isnan(temperature).any() or np.isnan(rh).any():
        raise ValueError("Freiburg gap-filled tensors still contain missing values")

    stats = pd.read_csv(stats_path)
    normalized = {
        str(column).lstrip("\ufeff").strip().casefold(): column for column in stats.columns
    }
    required = {
        "station_id": "station_id",
        "latitude_degn": "latitude_degN",
        "longitude_dege": "longitude_degE",
        "elevation_masl": "elevation_masl",
    }
    rename = {
        normalized[key]: canonical for key, canonical in required.items() if key in normalized
    }
    stats = stats.rename(columns=rename)
    missing = sorted(set(required.values()) - set(stats.columns))
    if missing:
        raise ValueError(
            "Freiburg statistics schema mismatch; "
            f"missing {missing}; observed columns={list(map(str, stats.columns))}"
        )

    stats = stats.set_index("station_id").reindex(station_ids)
    lat = pd.to_numeric(stats["latitude_degN"], errors="coerce").to_numpy(np.float32)
    lon = pd.to_numeric(stats["longitude_degE"], errors="coerce").to_numpy(np.float32)
    elevation = pd.to_numeric(stats["elevation_masl"], errors="coerce").to_numpy(np.float32)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or not np.isfinite(elevation).all():
        raise ValueError("Freiburg station metadata contains missing coordinates/elevation")

    edge_index, edge_attr = knn_graph(lat, lon, k)
    return UrbanDataset(
        name="freiburg",
        timestamps=index.to_numpy(),
        temperature=temperature,
        rh=rh,
        observed_mask=observed,
        station_ids=station_ids,
        lat=lat,
        lon=lon,
        elevation=elevation,
        edge_index=edge_index,
        edge_attr=edge_attr,
        source=SOURCE,
    )
