from __future__ import annotations

import argparse
import json
from pathlib import Path

from coolworld.config import Settings
from coolworld.evidence import EvidenceStore
from coolworld.fortyguard import DateTimeRequest, FortyGuardClient, HeatmapRequest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect one real FortyGuard heatmap and persist immutable provenance."
    )
    p.add_argument("--aoi", required=True, help="GeoJSON FeatureCollection/Feature polygon file")
    p.add_argument("--date", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--granularity", type=int, choices=(60, 80, 100), default=100)
    p.add_argument(
        "--analytic-type",
        choices=("tcm", "time_of_measure", "exceedance", "persistence"),
        default="tcm",
    )
    p.add_argument("--threshold", type=float)
    p.add_argument("--direction", choices=("above", "below"))
    a = p.parse_args()
    settings = Settings()
    if not settings.has_fortyguard_key:
        raise SystemExit("FORTYGUARD_API_KEY is missing from local .env/environment")
    aoi = json.loads(Path(a.aoi).read_text(encoding="utf-8"))
    req = HeatmapRequest(
        polygon_aoi=aoi,
        date_time=DateTimeRequest(start_date=a.date, start_time=a.time, filter_type=1),
        granularity=a.granularity,
        analytic_type=a.analytic_type,
        threshold=a.threshold,
        direction=a.direction,
    )
    result = FortyGuardClient(settings, EvidenceStore(settings.evidence_dir)).heatmap(req)
    print(
        json.dumps(
            {
                "activity_id": result.activity_id,
                "content_sha256": result.provenance.content_sha256,
                "request_sha256": result.provenance.request_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
