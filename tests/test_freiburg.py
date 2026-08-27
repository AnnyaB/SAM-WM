from pathlib import Path

import pandas as pd
import pytest

from coolworld.freiburg import OFFICIAL_COLUMNS, read_freiburg_table


def test_released_header_with_five_field_rows_is_reconstructed(tmp_path: Path):
    path = tmp_path / "freiburg.csv"
    path.write_text(
        'datetime_UTC,station_id,"variable,value",data_type\n'
        '2022-09-01T00:00:00Z,FRABCD,Ta_degC,20.5,observed\n'
        '2022-09-01T00:00:00Z,FRABCD,RH_percent,55.0,imputed\n',
        encoding="utf-8",
    )

    table = read_freiburg_table(path)

    assert tuple(table.columns) == OFFICIAL_COLUMNS
    assert table["station_id"].tolist() == ["FRABCD", "FRABCD"]
    assert table["variable"].tolist() == ["Ta_degC", "RH_percent"]
    assert table["value"].tolist() == [20.5, 55.0]
    assert table["data_type"].tolist() == ["observed", "imputed"]
    assert pd.api.types.is_datetime64_ns_dtype(table["datetime_UTC"])


def test_official_five_column_header_is_accepted(tmp_path: Path):
    path = tmp_path / "freiburg.csv"
    path.write_text(
        "datetime_UTC,station_id,variable,value,data_type\n"
        "2022-09-01T00:00:00Z,FRABCD,Ta_degC,20.5,observed\n",
        encoding="utf-8",
    )

    table = read_freiburg_table(path)
    assert tuple(table.columns) == OFFICIAL_COLUMNS
    assert table.iloc[0]["value"] == 20.5


def test_unknown_physical_layout_fails_closed(tmp_path: Path):
    path = tmp_path / "freiburg.csv"
    path.write_text(
        'datetime_UTC,station_id,"variable,value",data_type\n'
        "2022-09-01T00:00:00Z,FRABCD,Ta_degC,observed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="physical CSV layout mismatch"):
        read_freiburg_table(path)
