from pathlib import Path

from coolworld.timeline import load_recorded_heatmap_timeline


def test_empty_timeline_returns_empty(tmp_path: Path) -> None:
    assert load_recorded_heatmap_timeline(tmp_path) == []
