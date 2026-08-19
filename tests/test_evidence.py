from pathlib import Path

import pytest

from coolworld.evidence import EvidenceKind, EvidenceStore, Provenance, digest_json


def test_fortyguard_provenance_requires_real_activity_id(tmp_path: Path) -> None:
    p = Provenance(
        kind=EvidenceKind.FORTYGUARD_LIVE,
        source_name="FortyGuard",
        source_reference="https://api.fortyguard.com/v1/status/x",
        retrieved_at_utc="2026-08-18T00:00:00+00:00",
        content_sha256="0" * 64,
        request_sha256="1" * 64,
        activity_id=None,
    )
    with pytest.raises(ValueError):
        p.validate()


def test_content_addressed_store_rejects_hash_mismatch(tmp_path: Path) -> None:
    payload = {"measured": 41.0}
    p = Provenance(
        kind=EvidenceKind.CITY_OPEN_DATA,
        source_name="official city source",
        source_reference="https://example.invalid/not-used-in-test",
        retrieved_at_utc="2026-08-18T00:00:00+00:00",
        content_sha256="f" * 64,
    )
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path).persist_json(payload, p)


def test_digest_is_canonical_for_key_order() -> None:
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
