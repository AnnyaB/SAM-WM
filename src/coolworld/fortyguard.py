from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .evidence import atomic_json, digest_json


@dataclass(frozen=True)
class FortyGuardResult:
    activity_id: str
    payload: dict[str, Any]
    request_sha256: str
    content_sha256: str


class FortyGuardClient:
    """Minimal fail-closed, crash-resumable FortyGuard Temperature API client.

    A request intent is persisted before POST. Once an activity id is known, every
    restart resumes that exact provider activity instead of posting again. An
    ambiguous POST transport failure is never retried automatically because the
    server may already have accepted the request.
    """

    def __init__(
        self,
        evidence_dir: str | Path = "artifacts/fortyguard",
        base_url: str = "https://api.fortyguard.com",
    ) -> None:
        self.key = os.getenv("FORTYGUARD_API_KEY", "")
        if not self.key:
            raise RuntimeError("FORTYGUARD_API_KEY is not configured")
        self.base_url = base_url.rstrip("/")
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        (self.evidence_dir / "responses").mkdir(parents=True, exist_ok=True)

    def _load_registry(self, path: Path, request_hash: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("request_sha256") != request_hash:
            raise RuntimeError(f"FORTYGUARD_REQUEST_REGISTRY_HASH_MISMATCH: {path}")
        return state

    def heatmap(self, payload: dict[str, Any], *, max_wait_s: int = 300) -> FortyGuardResult:
        request_hash = digest_json(payload)
        registry = self.evidence_dir / f"{request_hash}.activity.json"
        state = self._load_registry(registry, request_hash)

        if state and state.get("state") == "COMPLETED":
            activity_id = str(state.get("activity_id", ""))
            content_hash = str(state.get("content_sha256", ""))
            response_path = self.evidence_dir / "responses" / f"{content_hash}.json"
            if activity_id and len(content_hash) == 64 and response_path.exists():
                body = json.loads(response_path.read_text(encoding="utf-8"))
                if digest_json(body) != content_hash:
                    raise RuntimeError("FORTYGUARD_CACHED_RESPONSE_HASH_MISMATCH")
                return FortyGuardResult(activity_id, body, request_hash, content_hash)
            raise RuntimeError("FORTYGUARD_COMPLETED_REGISTRY_MISSING_VERIFIED_RESPONSE")

        if state and state.get("state") == "AMBIGUOUS_POST_REQUIRES_REVIEW":
            raise RuntimeError(
                "FORTYGUARD_AMBIGUOUS_POST_REQUIRES_REVIEW: provider may have accepted "
                "the request; refusing automatic repost"
            )

        activity_id = str(state.get("activity_id", "")) if state else ""
        headers = {"api-key": self.key, "Content-Type": "application/json"}

        if not activity_id:
            atomic_json(
                registry,
                {
                    "request_sha256": request_hash,
                    "state": "INTENT_TO_SUBMIT",
                },
            )
            try:
                with httpx.Client(timeout=45) as client:
                    response = client.post(
                        f"{self.base_url}/v1/heatmap",
                        headers=headers,
                        json=payload,
                    )
            except httpx.RequestError as exc:
                atomic_json(
                    registry,
                    {
                        "request_sha256": request_hash,
                        "state": "AMBIGUOUS_POST_REQUIRES_REVIEW",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise RuntimeError(
                    "FortyGuard POST outcome is ambiguous; refusing automatic retry"
                ) from exc

            if response.status_code >= 400:
                atomic_json(
                    registry,
                    {
                        "request_sha256": request_hash,
                        "state": "SUBMIT_REJECTED",
                        "http_status": response.status_code,
                        "response_excerpt": response.text[:1000],
                    },
                )
                response.raise_for_status()

            body = response.json()
            activity_id = str(body.get("data", {}).get("activity_id", ""))
            if not activity_id:
                atomic_json(
                    registry,
                    {
                        "request_sha256": request_hash,
                        "state": "SUBMIT_RESPONSE_WITHOUT_ACTIVITY_ID",
                    },
                )
                raise RuntimeError("FortyGuard submit response did not contain activity_id")
            atomic_json(
                registry,
                {
                    "request_sha256": request_hash,
                    "activity_id": activity_id,
                    "state": "SUBMITTED_PENDING",
                },
            )

        deadline = time.monotonic() + max_wait_s
        delay = 4.0
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=45) as client:
                    response = client.get(
                        f"{self.base_url}/v1/status/{activity_id}",
                        headers={"api-key": self.key},
                    )
            except httpx.RequestError:
                time.sleep(delay)
                delay = min(delay * 1.5, 20.0)
                continue

            if response.status_code in {404, 429, 502, 503, 504}:
                time.sleep(delay)
                delay = min(delay * 1.5, 20.0)
                continue
            response.raise_for_status()
            body = response.json()
            status = str(body.get("data", {}).get("status", "")).lower()

            if status in {"failed", "error"}:
                atomic_json(
                    registry,
                    {
                        "request_sha256": request_hash,
                        "activity_id": activity_id,
                        "state": "PROVIDER_FAILED",
                    },
                )
                raise RuntimeError(f"FortyGuard activity failed: {activity_id}")

            if status in {"completed", "succeeded"}:
                content_hash = digest_json(body)
                response_path = self.evidence_dir / "responses" / f"{content_hash}.json"
                atomic_json(response_path, body)
                atomic_json(
                    registry,
                    {
                        "request_sha256": request_hash,
                        "activity_id": activity_id,
                        "content_sha256": content_hash,
                        "response_path": str(response_path),
                        "state": "COMPLETED",
                    },
                )
                return FortyGuardResult(activity_id, body, request_hash, content_hash)

            time.sleep(delay)
            delay = min(delay * 1.2, 15.0)

        raise TimeoutError(
            f"bounded polling exhausted for activity {activity_id}; registry preserved for resume"
        )
