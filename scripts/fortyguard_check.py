from __future__ import annotations

import argparse
import json

from coolworld.fortyguard import FortyGuardClient


def main() -> None:
    p = argparse.ArgumentParser(
        description="One real, auditable FortyGuard heatmap request. Never prints the API key."
    )
    p.add_argument("--date", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--aoi", required=True, help="GeoJSON FeatureCollection file")
    p.add_argument("--granularity", type=int, choices=[60, 80, 100], default=100)
    p.add_argument("--out", default="artifacts/fortyguard")
    a = p.parse_args()
    with open(a.aoi, encoding="utf-8") as handle:
        aoi = json.load(handle)
    payload = {
        "polygon_aoi": aoi,
        "date_time": {"start_date": a.date, "start_time": a.time, "filter_type": 1},
        "granularity": a.granularity,
        "analytic_type": "tcm",
    }
    r = FortyGuardClient(a.out).heatmap(payload)
    result = r.payload.get("data", {}).get("result", {})
    print(
        json.dumps(
            {
                "activity_id": r.activity_id,
                "request_sha256": r.request_sha256,
                "content_sha256": r.content_sha256,
                "has_map_data": bool(result.get("map_data")),
                "has_stats_data": bool(result.get("stats_data")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
