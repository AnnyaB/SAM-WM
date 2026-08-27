from pathlib import Path

import pandas as pd

from coolworld.benchmarks import _canonicalize_freiburg_table, _parse_sef_file


def test_freiburg_long_schema_is_case_and_whitespace_robust():
    raw = pd.DataFrame(
        {
            " datetime_UTC ": ["2022-09-01T00:00:00Z", "2022-09-01T00:00:00Z"],
            "station_id": ["FRABCD", "FRABCD"],
            "Variable": ["Ta_degC", "RH_percent"],
            "Value": [20.0, 55.0],
            "data_type": ["Observed", "IMPUTED"],
        }
    )
    table = _canonicalize_freiburg_table(raw)
    assert list(table.columns) == [
        "datetime_UTC",
        "station_id",
        "variable",
        "value",
        "data_type",
    ]
    assert table["variable"].tolist() == ["Ta_degC", "RH_percent"]
    assert table["data_type"].tolist() == ["observed", "imputed"]


def test_freiburg_released_combined_variable_value_field_is_split_strictly():
    raw = pd.DataFrame(
        {
            "datetime_UTC": ["2022-09-01T00:00:00Z", "2022-09-01T00:00:00Z"],
            "station_id": ["FRABCD", "FRABCD"],
            "variable,value": ["Ta_degC,20.0", "RH_percent,55.0"],
            "data_type": ["observed", "imputed"],
        }
    )
    table = _canonicalize_freiburg_table(raw)
    assert table["variable"].tolist() == ["Ta_degC", "RH_percent"]
    assert table["value"].tolist() == [20.0, 55.0]
    assert table["data_type"].tolist() == ["observed", "imputed"]


def test_freiburg_wide_schema_is_normalized_to_published_long_contract():
    raw = pd.DataFrame(
        {
            "datetime_UTC": ["2022-09-01T00:00:00Z"],
            "station_id": ["FRABCD"],
            "Ta_degC": [20.0],
            "RH_percent": [55.0],
            "data_type": ["observed"],
        }
    )
    table = _canonicalize_freiburg_table(raw)
    assert table.shape == (2, 5)
    assert set(table["variable"]) == {"Ta_degC", "RH_percent"}
    assert table["value"].tolist() == [20.0, 55.0]


def test_fairurbtemp_qc_flag_is_not_scored_as_ground_truth(tmp_path: Path):
    path = tmp_path / "station_hourly.tsv"
    path.write_text(
        "SEF\t1.0\n"
        "ID\tTEST01\n"
        "Lat\t48.0000\n"
        "Lon\t7.8000\n"
        "Year\tMonth\tDay\tHour\tMinute\tValue\tMeta\n"
        "2023\t7\t1\t12\t0\t30.0\t\n"
        "2023\t7\t1\t13\t0\t99.0\tqc = gross_error\n",
        encoding="utf-8",
    )
    station, lat, lon, series = _parse_sef_file(path)
    assert station == "TEST01"
    assert lat == 48.0
    assert lon == 7.8
    assert list(series.values) == [30.0]
