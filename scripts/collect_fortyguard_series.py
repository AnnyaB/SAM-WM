from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coolworld.config import Settings
from coolworld.evidence import EvidenceStore
from coolworld.fortyguard import DateTimeRequest, FortyGuardClient, HeatmapRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a declared schedule of real FortyGuard single-hour TCM heatmaps."
    )
    parser.add_argument("--aoi", required=True, help="GeoJSON polygon/FeatureCollection")
    parser.add_argument(
        "--schedule",
        required=True,
        help="CSV with exactly date,time columns; every row becomes one real API request",
    )
    parser.add_argument("--granularity", type=int, choices=(60, 80, 100), default=100)
    args = parser.parse_args()

    settings = Settings()
    if not settings.has_fortyguard_key:
        raise SystemExit("FORTYGUARD_API_KEY is missing from local .env/environment")
    aoi = json.loads(Path(args.aoi).read_text(encoding="utf-8"))
    with Path(args.schedule).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"date", "time"}:
        raise SystemExit("schedule CSV must contain exactly: date,time")

    client = FortyGuardClient(settings, EvidenceStore(settings.evidence_dir))
    completed: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        request = HeatmapRequest(
            polygon_aoi=aoi,
            date_time=DateTimeRequest(
                start_date=row["date"], start_time=row["time"], filter_type=1
            ),
            granularity=args.granularity,
            analytic_type="tcm",
        )
        result = client.heatmap(request)
        record = {
            "date": row["date"],
            "time": row["time"],
            "activity_id": result.activity_id,
            "content_sha256": result.provenance.content_sha256,
            "request_sha256": result.provenance.request_sha256 or "",
        }
        completed.append(record)
        print(json.dumps({"index": i, "total": len(rows), **record}))

    print(json.dumps({"completed": len(completed), "records": completed}, indent=2))


if __name__ == "__main__":
    main()
