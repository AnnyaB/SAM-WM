from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .evidence import (
    EvidenceKind,
    EvidenceStore,
    Provenance,
    digest_bytes,
    digest_json,
    utc_now_iso,
)

BASE_URL = "https://api.fortyguard.com"


class DateTimeRequest(BaseModel):
    start_date: str
    filter_type: int = Field(ge=1, le=4)
    start_time: str | None = None
    end_time: str | None = None
    end_date: str | None = None

    @field_validator("start_date")
    @classmethod
    def hackathon_date_floor(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed < date(2021, 1, 1):
            raise ValueError("hackathon evidence must be dated 2021-01-01 or later")
        return value


class HeatmapRequest(BaseModel):
    polygon_aoi: dict[str, Any]
    date_time: DateTimeRequest
    granularity: int = 100
    analytic_type: str = "tcm"
    threshold: float | None = None
    direction: str | None = None

    @field_validator("granularity")
    @classmethod
    def valid_granularity(cls, value: int) -> int:
        if value not in {60, 80, 100}:
            raise ValueError("FortyGuard heatmap granularity must be 60, 80, or 100")
        return value


@dataclass(frozen=True, slots=True)
class LiveResult:
    activity_id: str
    payload: dict[str, Any]
    provenance: Provenance


class FortyGuardError(RuntimeError):
    pass


class FortyGuardClient:
    """Thin client matching the documented submit -> activity_id -> poll flow."""

    def __init__(self, settings: Settings, store: EvidenceStore) -> None:
        if not settings.has_fortyguard_key:
            raise FortyGuardError("FORTYGUARD_API_KEY is not configured")
        self._key = settings.fortyguard_api_key.get_secret_value()  # type: ignore[union-attr]
        self._timeout = settings.http_timeout_seconds
        self._store = store

    @property
    def _headers(self) -> dict[str, str]:
        return {"api-key": self._key, "Content-Type": "application/json"}

    def submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        request_hash = digest_json(payload)
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{BASE_URL}{endpoint}", headers=self._headers, json=payload)
        response.raise_for_status()
        body = response.json()
        try:
            activity_id = str(body["data"]["activity_id"])
        except (KeyError, TypeError) as exc:
            raise FortyGuardError("FortyGuard response did not contain activity_id") from exc
        if not activity_id:
            raise FortyGuardError("FortyGuard returned an empty activity_id")
        # Persist the secret-free request body so later dataset construction can
        # recover its timestamp/AOI without relying on memory or notebook state.
        self._store.persist_request_json(payload, request_hash)
        return activity_id

    def status(self, activity_id: str) -> dict[str, Any]:
        if not activity_id.strip():
            raise ValueError("activity_id must not be empty")
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                f"{BASE_URL}/v1/status/{activity_id}",
                headers={"api-key": self._key},
            )
        response.raise_for_status()
        return response.json()

    def wait_for_json(
        self,
        *,
        activity_id: str,
        original_request: dict[str, Any],
        poll_seconds: float = 5.0,
        timeout_seconds: float = 300.0,
    ) -> LiveResult:
        deadline = time.monotonic() + timeout_seconds
        while True:
            body = self.status(activity_id)
            status = str(body.get("data", {}).get("status", "")).lower()
            if status in {"completed", "succeeded"}:
                provenance = Provenance(
                    kind=EvidenceKind.FORTYGUARD_LIVE,
                    source_name="FortyGuard Temperature API",
                    source_reference=f"{BASE_URL}/v1/status/{activity_id}",
                    retrieved_at_utc=utc_now_iso(),
                    content_sha256=digest_json(body),
                    request_sha256=digest_json(original_request),
                    activity_id=activity_id,
                )
                self._store.persist_json(body, provenance)
                return LiveResult(activity_id=activity_id, payload=body, provenance=provenance)
            if status in {"failed", "error"}:
                raise FortyGuardError(f"FortyGuard activity {activity_id} failed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FortyGuard activity {activity_id} did not finish in time")
            time.sleep(poll_seconds)

    def heatmap(self, request: HeatmapRequest) -> LiveResult:
        payload = request.model_dump(exclude_none=True)
        activity_id = self.submit("/v1/heatmap", payload)
        return self.wait_for_json(activity_id=activity_id, original_request=payload)

    def env_params(self, request: EnvironmentalParametersRequest) -> LiveResult:
        payload = request.model_dump(exclude_none=True)
        activity_id = self.submit("/v1/env_params", payload)
        return self.wait_for_json(activity_id=activity_id, original_request=payload)

    def satellite(self, request: SatelliteRequest) -> LiveResult:
        payload = request.model_dump(exclude_none=True)
        activity_id = self.submit("/v1/satellite", payload)
        return self.wait_for_json(activity_id=activity_id, original_request=payload)

    def streetview(self, request: StreetViewRequest) -> LiveResult:
        payload = request.model_dump(exclude_none=True)
        activity_id = self.submit("/v1/streetview", payload)
        return self.wait_for_json(activity_id=activity_id, original_request=payload)

    def heat_intelligence_pdf(
        self,
        request: HeatIntelligenceRequest,
        *,
        poll_seconds: float = 5.0,
        timeout_seconds: float = 300.0,
    ) -> tuple[str, bytes, Provenance]:
        payload = request.model_dump(exclude_none=True)
        activity_id = self.submit("/v1/heat_intelligence", payload)
        deadline = time.monotonic() + timeout_seconds
        while True:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    f"{BASE_URL}/v1/status/{activity_id}", headers={"api-key": self._key}
                )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type:
                raw = response.content
                provenance = Provenance(
                    kind=EvidenceKind.FORTYGUARD_LIVE,
                    source_name="FortyGuard Heat Intelligence",
                    source_reference=f"{BASE_URL}/v1/status/{activity_id}",
                    retrieved_at_utc=utc_now_iso(),
                    content_sha256=digest_bytes(raw),
                    request_sha256=digest_json(payload),
                    activity_id=activity_id,
                )
                self._store.persist_bytes(raw, provenance, suffix=".pdf")
                return activity_id, raw, provenance
            try:
                body = response.json()
            except ValueError as exc:
                raise FortyGuardError(
                    "Heat Intelligence status returned neither JSON nor PDF"
                ) from exc
            status = str(body.get("data", {}).get("status", "")).lower()
            if status in {"failed", "error"}:
                raise FortyGuardError(f"FortyGuard activity {activity_id} failed")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FortyGuard activity {activity_id} did not finish in time")
            time.sleep(poll_seconds)


class EnvironmentalParametersRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    temperature: float
    date_time: DateTimeRequest


class SatelliteRequest(BaseModel):
    sat: dict[str, float]
    date_time: DateTimeRequest
    granularity: int = 100

    @field_validator("granularity")
    @classmethod
    def valid_satellite_granularity(cls, value: int) -> int:
        if value not in {60, 80, 100}:
            raise ValueError("FortyGuard satellite granularity must be 60, 80, or 100")
        return value

    @field_validator("sat")
    @classmethod
    def valid_satellite_location(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"latitude", "longitude"}:
            raise ValueError("sat must contain exactly latitude and longitude")
        lat, lon = float(value["latitude"]), float(value["longitude"])
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("invalid satellite coordinates")
        return {"latitude": lat, "longitude": lon}


class StreetViewRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    vertical_angle: float
    horizontal_angle: float = Field(ge=0, le=360)
    back_view: bool = False


class HeatIntelligenceRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    temperature: float
    date: str
    analysis: list[str]

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed < date(2021, 1, 1):
            raise ValueError("hackathon evidence must be dated 2021-01-01 or later")
        return value

    @field_validator("analysis")
    @classmethod
    def validate_analysis(cls, value: list[str]) -> list[str]:
        allowed = {"geographic", "environmental", "urban", "events", "anthropogenic"}
        if not value or any(x not in allowed for x in value):
            raise ValueError(f"analysis values must be chosen from {sorted(allowed)}")
        return value
