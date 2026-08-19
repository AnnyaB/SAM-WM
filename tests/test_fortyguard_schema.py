import pytest
from pydantic import ValidationError

from coolworld.fortyguard import HeatmapRequest


def test_heatmap_rejects_undocumented_granularity() -> None:
    with pytest.raises(ValidationError):
        HeatmapRequest(
            polygon_aoi={"type": "FeatureCollection", "features": []},
            date_time={"start_date": "2026-08-18", "start_time": "14:00", "filter_type": 1},
            granularity=50,
        )


def test_hackathon_date_before_2021_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HeatmapRequest(
            polygon_aoi={"type": "FeatureCollection", "features": []},
            date_time={"start_date": "2020-12-31", "start_time": "14:00", "filter_type": 1},
            granularity=100,
        )
