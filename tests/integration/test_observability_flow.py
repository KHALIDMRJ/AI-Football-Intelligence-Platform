"""
Integration coverage for the observability layer end-to-end.

Exercises the full middleware → metrics → readiness-probe stack through
the FastAPI app so the pieces prove they compose:

* ``/health`` stamps and echoes ``X-Request-ID`` on the response.
* An incoming ``X-Request-ID`` is preserved rather than overwritten.
* ``/metrics`` returns a Prometheus exposition that reflects traffic
  the test itself just produced.
* ``/health/ready`` returns 200 when DB + cache are both reachable
  (the in-memory fixtures always are).
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_stamps_request_id_on_response(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    rid = resp.headers.get("X-Request-ID")
    assert rid and len(rid) >= 16


@pytest.mark.asyncio
async def test_incoming_request_id_is_preserved(client):
    resp = await client.get("/health", headers={"X-Request-ID": "trace-xyz-999"})
    assert resp.headers["X-Request-ID"] == "trace-xyz-999"


@pytest.mark.asyncio
async def test_metrics_endpoint_reflects_traffic(client):
    # Generate at least one matched request so the counter has a sample.
    await client.get("/health")
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "football_ai_http_requests_total" in body
    assert "football_ai_http_request_duration_seconds" in body
    # The /health call above should have been labelled and counted.
    assert 'route="/health"' in body


@pytest.mark.asyncio
async def test_readiness_probe_reports_ok_in_tests(client):
    resp = await client.get("/health/ready")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["database"]["ok"] is True
    assert payload["checks"]["cache"]["ok"] is True
