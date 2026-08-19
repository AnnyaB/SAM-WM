from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from .config import Settings
from .evidence import EvidenceKind, EvidenceStore, Provenance, digest_json, utc_now_iso


@dataclass(frozen=True, slots=True)
class PhoenixLayer:
    name: str
    layer_url: str


COOL_PAVEMENT = PhoenixLayer(
    name="City of Phoenix Cool Pavement Projects",
    layer_url=("https://maps.phoenix.gov/pub/rest/services/Public/STR_MainCoolPave/MapServer/1"),
)

COOL_CORRIDORS = PhoenixLayer(
    name="City of Phoenix Cool Corridors",
    layer_url=("https://maps.phoenix.gov/pub/rest/services/Public/STR_CoolCorridors/MapServer/0"),
)

TREE_CANOPY = PhoenixLayer(
    name="City of Phoenix Tree Canopy by Census Tract",
    layer_url=(
        "https://maps.phoenix.gov/pub/rest/services/Public/Shade_Study_Data_CMO_OHR/MapServer/1"
    ),
)


class PhoenixOpenDataClient:
    def __init__(self, settings: Settings, store: EvidenceStore) -> None:
        self._timeout = settings.http_timeout_seconds
        self._store = store

    def fetch_geojson(self, layer: PhoenixLayer, where: str = "1=1") -> dict[str, Any]:
        parsed = urlparse(layer.layer_url)
        if parsed.scheme != "https" or parsed.hostname != "maps.phoenix.gov":
            raise ValueError("Phoenix connector only permits the official maps.phoenix.gov host")

        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        query_url = f"{layer.layer_url}/query?{urlencode(params)}"
        with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
            response = client.get(query_url)
        response.raise_for_status()
        payload = response.json()
        if payload.get("type") != "FeatureCollection":
            raise RuntimeError("official Phoenix service did not return GeoJSON FeatureCollection")

        provenance = Provenance(
            kind=EvidenceKind.CITY_OPEN_DATA,
            source_name=layer.name,
            source_reference=query_url,
            retrieved_at_utc=utc_now_iso(),
            content_sha256=digest_json(payload),
        )
        self._store.persist_json(payload, provenance)
        return {
            "data": payload,
            "provenance": {
                "kind": provenance.kind.value,
                "source_name": provenance.source_name,
                "source_reference": provenance.source_reference,
                "retrieved_at_utc": provenance.retrieved_at_utc,
                "content_sha256": provenance.content_sha256,
            },
        }
