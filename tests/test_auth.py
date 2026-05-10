"""Tests for the API-key authentication dependency.

Covers:
- Open mode (no API key configured): all requests pass through.
- Protected mode (API key set): valid key accepted, wrong key rejected,
  missing header rejected.
- Auth is enforced on POST /index and DELETE /index.
- Auth is NOT required on read-only endpoints (GET /health, GET /datasets).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

_TEST_KEY = "test-secret-key-abc123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(tmp_path: Path, api_key: str = "") -> object:
    """Create a fresh app with an isolated settings + registry state."""
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    root_dir = tmp_path / "data"
    root_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={root_dir}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    if api_key:
        os.environ["ARRO_SERVER_API_KEY"] = api_key
    else:
        os.environ.pop("ARRO_SERVER_API_KEY", None)

    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()

    return create_app()


def _cleanup() -> None:
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.storage import registry as registry_mod

    for key in ("ARRO_SERVER_DATA_ROOTS", "ARRO_SERVER_SERVE_FRONTEND", "ARRO_SERVER_API_KEY"):
        os.environ.pop(key, None)
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


# ---------------------------------------------------------------------------
# Open mode (no key configured)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_open_mode(tmp_path):
    app = _make_app(tmp_path)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is False
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_post_index_open_mode_no_header(tmp_path):
    """Without an API key configured, POST /index requires no auth header."""
    app = _make_app(tmp_path)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # We just check auth passes (404 = dataset not found, not 401)
            r = await c.post("/api/datasets/main--nonexistent/index")
        assert r.status_code != 401
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_delete_index_open_mode_no_header(tmp_path):
    app = _make_app(tmp_path)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/api/datasets/main--nonexistent/index")
        assert r.status_code != 401
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Protected mode (key configured)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_protected_mode_no_auth_required(tmp_path):
    """GET /health is always public — no key needed even in protected mode."""
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["auth_enabled"] is True
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_post_index_missing_key_returns_401(tmp_path):
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/datasets/main--matrix/index")
        assert r.status_code == 401
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_post_index_wrong_key_returns_401(tmp_path):
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/datasets/main--matrix/index",
                headers={"X-API-Key": "wrong-key"},
            )
        assert r.status_code == 401
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_post_index_correct_key_passes_auth(tmp_path):
    """Correct key → auth passes; 404 because dataset doesn't exist — not 401."""
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/datasets/main--matrix/index",
                headers={"X-API-Key": _TEST_KEY},
            )
        assert r.status_code != 401
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_delete_index_missing_key_returns_401(tmp_path):
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete("/api/datasets/main--matrix/index")
        assert r.status_code == 401
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_delete_index_correct_key_passes_auth(tmp_path):
    """Correct key → auth passes; returns 200 with deleted=False (no index built)."""
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.delete(
                "/api/datasets/main--matrix/index",
                headers={"X-API-Key": _TEST_KEY},
            )
        assert r.status_code == 200
        assert r.json()["deleted"] is False
    finally:
        _cleanup()


@pytest.mark.asyncio
async def test_read_endpoints_never_require_auth(tmp_path):
    """Read endpoints (GET /datasets, GET /health) are always public."""
    app = _make_app(tmp_path, api_key=_TEST_KEY)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for path in ("/api/health", "/api/datasets"):
                r = await c.get(path)
                assert r.status_code not in (401, 403), f"{path} should be public"
    finally:
        _cleanup()
