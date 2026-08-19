import os

import pytest

from coolworld.config import Settings
from coolworld.evidence import EvidenceStore
from coolworld.phoenix import COOL_PAVEMENT, PhoenixOpenDataClient


@pytest.mark.live
def test_official_phoenix_cool_pavement_is_retrievable(tmp_path) -> None:
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("set RUN_LIVE_TESTS=1 to execute real network integration test")
    result = PhoenixOpenDataClient(Settings(), EvidenceStore(tmp_path)).fetch_geojson(COOL_PAVEMENT)
    assert result["data"]["type"] == "FeatureCollection"
    assert len(result["data"]["features"]) > 0
