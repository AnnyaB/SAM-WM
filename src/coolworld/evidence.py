from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any


class EvidenceKind(StrEnum):
    FORTYGUARD_LIVE = "fortyguard_live"
    FORTYGUARD_RECORDED_LIVE = "fortyguard_recorded_live"
    CITY_OPEN_DATA = "city_open_data"
    REAL_INTERVENTION_STUDY = "real_intervention_study"


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: EvidenceKind
    source_name: str
    source_reference: str
    retrieved_at_utc: str
    content_sha256: str
    request_sha256: str | None = None
    activity_id: str | None = None

    def validate(self) -> None:
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 hex digest")
        if self.kind in {EvidenceKind.FORTYGUARD_LIVE, EvidenceKind.FORTYGUARD_RECORDED_LIVE}:
            if not self.activity_id:
                raise ValueError("FortyGuard evidence requires a real activity_id")
            if not self.request_sha256:
                raise ValueError("FortyGuard evidence requires request_sha256")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json_bytes(value))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class EvidenceStore:
    """Append-only-ish evidence store.

    Files are content-addressed. Existing content is never silently overwritten
    with different bytes because the SHA-256 is part of the filename.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def persist_json(self, payload: Any, provenance: Provenance) -> tuple[Path, Path]:
        provenance.validate()
        raw = canonical_json_bytes(payload)
        if digest_bytes(raw) != provenance.content_sha256:
            raise ValueError("payload does not match provenance content hash")

        folder = self.root / provenance.kind.value
        folder.mkdir(parents=True, exist_ok=True)
        stem = provenance.content_sha256
        data_path = folder / f"{stem}.json"
        provenance_payload = asdict(provenance)
        provenance_hash = digest_json(provenance_payload)
        meta_path = folder / f"{stem}.{provenance_hash[:16]}.provenance.json"

        if data_path.exists() and data_path.read_bytes() != raw:
            raise RuntimeError("hash collision or evidence corruption detected")
        if not data_path.exists():
            data_path.write_bytes(raw)

        meta_raw = json.dumps(provenance_payload, indent=2, sort_keys=True)
        if meta_path.exists() and meta_path.read_text(encoding="utf-8") != meta_raw:
            raise RuntimeError("provenance record changed after creation")
        if not meta_path.exists():
            meta_path.write_text(meta_raw, encoding="utf-8")
        return data_path, meta_path

    def persist_request_json(self, payload: Any, expected_sha256: str) -> Path:
        """Persist a secret-free API request body by canonical content hash."""
        raw = canonical_json_bytes(payload)
        actual = digest_bytes(raw)
        if actual != expected_sha256:
            raise ValueError("request payload does not match expected SHA-256")
        folder = self.root / "requests"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{actual}.json"
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError("request hash collision or corruption detected")
        if not path.exists():
            path.write_bytes(raw)
        return path

    def persist_bytes(
        self, raw: bytes, provenance: Provenance, *, suffix: str
    ) -> tuple[Path, Path]:
        """Persist non-JSON evidence such as a streamed official PDF."""
        provenance.validate()
        if digest_bytes(raw) != provenance.content_sha256:
            raise ValueError("bytes do not match provenance content hash")
        clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        folder = self.root / provenance.kind.value
        folder.mkdir(parents=True, exist_ok=True)
        stem = provenance.content_sha256
        data_path = folder / f"{stem}{clean_suffix}"
        provenance_payload = asdict(provenance)
        provenance_hash = digest_json(provenance_payload)
        meta_path = folder / f"{stem}.{provenance_hash[:16]}.provenance.json"
        if data_path.exists() and data_path.read_bytes() != raw:
            raise RuntimeError("hash collision or evidence corruption detected")
        if not data_path.exists():
            data_path.write_bytes(raw)
        meta_raw = json.dumps(provenance_payload, indent=2, sort_keys=True)
        if not meta_path.exists():
            meta_path.write_text(meta_raw, encoding="utf-8")
        return data_path, meta_path
