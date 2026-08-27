from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from coolworld.fortyguard import FortyGuardClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a bounded consecutive FortyGuard TCM timeline with crash-resume evidence."
    )
    parser.add_argument("--start", required=True, help="Local/provider time, e.g. 2026-08-20T00:00")
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument("--aoi", required=True)
    parser.add_argument("--granularity", type=int, choices=[60, 80, 100], default=100)
    parser.add_argument("--evidence", default="artifacts/fortyguard")
    parser.add_argument("--confirm-credit-usage", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.hours <= 240:
        raise SystemExit("--hours must lie in [1,240]")
    start = datetime.fromisoformat(args.start)
    aoi = json.loads(Path(args.aoi).read_text(encoding="utf-8"))
    if aoi.get("type") != "FeatureCollection" or not aoi.get("features"):
        raise SystemExit("AOI must be a non-empty GeoJSON FeatureCollection")

    schedule = [start + timedelta(hours=offset) for offset in range(args.hours)]
    if not args.confirm_credit_usage:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_NO_API_CALLS",
                    "requests": len(schedule),
                    "first": schedule[0].isoformat(timespec="minutes"),
                    "last": schedule[-1].isoformat(timespec="minutes"),
                    "instruction": "re-run with --confirm-credit-usage after reviewing request count",
                },
                indent=2,
            )
        )
        return

    client = FortyGuardClient(args.evidence)
    completed = []
    for timestamp in schedule:
        payload = {
            "polygon_aoi": aoi,
            "date_time": {
                "start_date": timestamp.date().isoformat(),
                "start_time": timestamp.strftime("%H:%M"),
                "filter_type": 1,
            },
            "granularity": args.granularity,
            "analytic_type": "tcm",
        }
        result = client.heatmap(payload)
        completed.append(
            {
                "timestamp": timestamp.isoformat(timespec="minutes"),
                "activity_id": result.activity_id,
                "request_sha256": result.request_sha256,
                "content_sha256": result.content_sha256,
            }
        )
        print(json.dumps(completed[-1], sort_keys=True), flush=True)

    print(json.dumps({"status": "COMPLETE", "completed": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
