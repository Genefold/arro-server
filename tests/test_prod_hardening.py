"""Tests covering the production-hardening changes:

- DELETE /api/datasets/{id}/index  (idempotent, returns {deleted: bool})
- GET /api/health  includes indexed_datasets and auth_enabled keys
- GET /api/datasets/{id}/data  uses default_window when limit is omitted
- CORS middleware uses allow_methods that includes DELETE
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_data(tmp_zarr_root):
    """Reuse the tmp_zarr_root fixture from conftest and wire up a fresh app."""
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ.pop("ARRO_SERVER_API_KEY", None)  # open mode
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()
    app = create_app()
    yield app
    for k in ("ARRO_SERVER_DATA_ROOTS", "ARRO_SERVER_SERVE_FRONTEND", "ARRO_SERVER_API_KEY"):
        os.environ.pop(k, None)
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_has_indexed_datasets_key(app_with_data):
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r = await c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "indexed_datasets" in body
    assert isinstance(body["indexed_datasets"], list)


@pytest.mark.asyncio
async def test_health_has_auth_enabled_key(app_with_data):
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r = await c.get("/api/health")
    body = r.json()
    assert "auth_enabled" in body
    assert body["auth_enabled"] is False  # open mode in this fixture


# ---------------------------------------------------------------------------
# DELETE /index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_index_idempotent_returns_false_when_no_index(app_with_data):
    """Deleting a dataset that was never indexed returns deleted=False, not an error."""
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r = await c.delete("/api/datasets/main--matrix/index")
    assert r.status_code == 200
    assert r.json() == {"id": "main--matrix", "deleted": False}


@pytest.mark.asyncio
async def test_delete_index_repeated_calls_idempotent(app_with_data):
    """Two consecutive DELETE calls must both succeed (second returns deleted=False)."""
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r1 = await c.delete("/api/datasets/main--matrix/index")
        r2 = await c.delete("/api/datasets/main--matrix/index")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["deleted"] is False


# ---------------------------------------------------------------------------
# default_window fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_uses_default_window_when_limit_omitted(app_with_data):
    """GET /data without ?limit should return at most default_window rows."""
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r = await c.get("/api/datasets/main--matrix/data")
    assert r.status_code == 200
    body = r.json()
    # default_window defaults to 1000; matrix has 50 rows — all should be returned
    assert body["limit"] == 1000
    assert body["data"]["shape"][0] <= 1000


# ---------------------------------------------------------------------------
# CORS allow_methods includes DELETE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_preflight_delete_allowed(app_with_data):
    """OPTIONS pre-flight for DELETE should return 200 with DELETE in Allow header."""
    async with AsyncClient(transport=ASGITransport(app=app_with_data), base_url="http://test") as c:
        r = await c.options(
            "/api/datasets/main--matrix/index",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
            },
        )
    # CORS middleware returns 200 for valid pre-flights
    assert r.status_code == 200
