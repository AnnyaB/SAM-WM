from __future__ import annotations

import json

import httpx
import pytest

from coolworld.fortyguard import FortyGuardClient


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad response",
                request=httpx.Request("GET", "https://example.invalid"),
                response=httpx.Response(self.status_code),
            )


def test_completed_response_is_content_addressed_and_reused(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test-secret-not-real")
    calls = {"post": 0, "get": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            calls["post"] += 1
            return _Response(200, {"data": {"activity_id": "activity-1"}})

        def get(self, *args, **kwargs):
            calls["get"] += 1
            return _Response(
                200,
                {
                    "data": {
                        "status": "Completed",
                        "result": {"map_data": {"type": "FeatureCollection", "features": []}},
                    }
                },
            )

    monkeypatch.setattr("coolworld.fortyguard.httpx.Client", FakeClient)
    payload = {"granularity": 100, "date_time": {"start_date": "2026-08-26"}}
    client = FortyGuardClient(tmp_path)
    first = client.heatmap(payload, max_wait_s=1)
    assert len(first.content_sha256) == 64
    assert calls == {"post": 1, "get": 1}

    second = client.heatmap(payload, max_wait_s=1)
    assert second.content_sha256 == first.content_sha256
    assert calls == {"post": 1, "get": 1}


def test_ambiguous_post_is_never_automatically_reposted(monkeypatch, tmp_path):
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test-secret-not-real")
    calls = {"post": 0}

    class BrokenClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            calls["post"] += 1
            raise httpx.ReadTimeout("ambiguous timeout")

    monkeypatch.setattr("coolworld.fortyguard.httpx.Client", BrokenClient)
    payload = {"granularity": 100}
    client = FortyGuardClient(tmp_path)
    with pytest.raises(RuntimeError, match="ambiguous"):
        client.heatmap(payload, max_wait_s=1)
    assert calls["post"] == 1

    with pytest.raises(RuntimeError, match="AMBIGUOUS_POST"):
        client.heatmap(payload, max_wait_s=1)
    assert calls["post"] == 1
