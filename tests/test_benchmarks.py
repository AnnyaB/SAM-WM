from pathlib import Path

from coolworld.benchmarks import _parse_sef_file


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
