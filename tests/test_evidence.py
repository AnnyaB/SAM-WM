from __future__ import annotations

import hashlib
from pathlib import Path

from coolworld.evidence import digest_json, sha256_file


def test_digest_is_order_invariant():
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})


def test_sha256_file_streams_exact_bytes(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    payload = b"sam-wm\x00evidence\n" * 1024
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
