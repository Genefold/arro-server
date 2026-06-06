from __future__ import annotations

import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(configured_app):
    with TestClient(configured_app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def client_with_token(tmp_zarr_root):
    import os

    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_ADMIN_TOKEN"] = "secret-token"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()  # full singleton reset for test isolation
    arrowspace_adapter.reset_adapter_cache()
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    for key in (
        "ARRO_SERVER_DATA_ROOTS",
        "ARRO_SERVER_SERVE_FRONTEND",
        "ARRO_SERVER_ADMIN_TOKEN",
    ):
        os.environ.pop(key, None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()  # full singleton reset for test isolation
    arrowspace_adapter.reset_adapter_cache()


# ---------------------------------------------------------------------------
# Basic reload
# ---------------------------------------------------------------------------


def test_reload_returns_200(client):
    r = client.post("/api/admin/reload")
    assert r.status_code == 200


def test_reload_response_shape(client):
    r = client.post("/api/admin/reload")
    body = r.json()
    assert body["reloaded"] is True
    assert isinstance(body["datasets_found"], int)
    assert isinstance(body["data_roots"], list)
    assert isinstance(body["indexed_datasets"], list)


def test_reload_datasets_found_matches_list(client):
    datasets_before = client.get("/api/datasets").json()["datasets"]
    r = client.post("/api/admin/reload")
    assert r.json()["datasets_found"] == len(
        [d for d in datasets_before if d["kind"] == "array"]
    )


def test_reload_is_idempotent(client):
    r1 = client.post("/api/admin/reload")
    r2 = client.post("/api/admin/reload")
    assert r1.json()["datasets_found"] == r2.json()["datasets_found"]


# ---------------------------------------------------------------------------
# New dataset becomes visible after reload
# ---------------------------------------------------------------------------


def test_reload_reports_new_datasets(client, tmp_zarr_root):
    """Reload picks up a Zarr array written after startup on a known root.

    Sequence:
      1. Record baseline dataset IDs from GET /api/datasets
      2. Write a new Zarr array directly to tmp_zarr_root (simulates arro-memory
         writing to a shared volume)
      3. Call POST /api/admin/reload
      4. Assert reload response datasets_found == baseline + 1
      5. Assert GET /api/datasets includes the new dataset
    """
    zarr = pytest.importorskip("zarr")
    import numpy as np

    # 1. Baseline
    before_ids = {d["id"] for d in client.get("/api/datasets").json()["datasets"]}
    before_count = len([
        d for d in client.get("/api/datasets").json()["datasets"]
        if d["kind"] == "array"
    ])

    # 2. Write new Zarr to the same root AFTER startup (simulates external write)
    new_path = tmp_zarr_root / "new_embeddings"
    arr = zarr.open(str(new_path), mode="w", shape=(20, 8), chunks=(10, 8), dtype="float64")
    arr[:] = np.random.default_rng(0).standard_normal((20, 8))

    # 3. Reload
    r = client.post("/api/admin/reload")
    assert r.status_code == 200

    # 4. reload response count is consistent
    assert r.json()["datasets_found"] == before_count + 1

    # 5. New dataset is now visible
    after_ids = {d["id"] for d in client.get("/api/datasets").json()["datasets"]}
    new_ids = after_ids - before_ids
    assert len(new_ids) == 1, f"Expected exactly 1 new dataset, got: {new_ids}"
    assert any("new_embeddings" in i for i in new_ids)


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_reload_no_auth_when_token_unset(client):
    r = client.post("/api/admin/reload")
    assert r.status_code == 200


def test_reload_401_when_token_set_and_missing(client_with_token):
    r = client_with_token.post("/api/admin/reload")
    assert r.status_code == 401


def test_reload_401_when_token_wrong(client_with_token):
    r = client_with_token.post(
        "/api/admin/reload",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401


def test_reload_200_when_token_correct(client_with_token):
    r = client_with_token.post(
        "/api/admin/reload",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert r.status_code == 200
    assert r.json()["reloaded"] is True
