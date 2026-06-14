"""Tests for GET /api/datasets/{dataset_id}/vectors/count.

Test index:
    1.  test_count_warm_cache              — O(1) path: dataset registered before request
    2.  test_count_cold_cache              — cache invalidated before request; lazy load triggered
    3.  test_count_dataset_not_found_404   — unknown dataset_id → 404
    4.  test_count_after_append            — nrows increments correctly after vectors/append
    5.  test_count_after_overwrite         — nrows unchanged after vectors/overwrite
    6.  test_count_shape_echoed            — dataset_id is echoed back in response
    7.  test_count_1d_array_422            — 1-D Zarr array → 422 (defensive guard)

Design notes:
    - All tests use TestClient (WSGI, synchronous) — no async fixtures needed.
    - The fixture registers one float32 (50, 8) array at main--counter.
    - Cache invalidation is tested by calling get_registry().invalidate()
      directly after app creation, before the first request.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client(tmp_path: Path):
    """TestClient backed by a single float32 Zarr array at main--counter.

    Shape (50, 8), dtype float32.
    All values are zero (np.zeros) — content is irrelevant for count tests.
    """
    import zarr

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    zarr_dir = tmp_path / "counter"
    arr = zarr.open(
        str(zarr_dir), mode="w", shape=(50, 8), chunks=(10, 8), dtype="float32"
    )
    arr[:] = np.zeros((50, 8), dtype="float32")

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_path}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1. Warm cache
# ---------------------------------------------------------------------------


def test_count_warm_cache(app_client):
    """Standard O(1) path: registry cache is warm, nrows=50, ncols=8."""
    client, _ = app_client
    # Warm the cache with a list call first.
    warmup = client.get("/api/datasets")
    assert warmup.status_code == 200

    resp = client.get("/api/datasets/main--counter/vectors/count")
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["nrows"] == 50
    assert body["ncols"] == 8
    assert body["dataset_id"] == "main--counter"


# ---------------------------------------------------------------------------
# 2. Cold cache — lazy load triggered
# ---------------------------------------------------------------------------


def test_count_cold_cache(app_client):
    """If cache is invalidated, the endpoint triggers a full rescan and succeeds."""
    client, _ = app_client
    from arro_server.storage.registry import get_registry

    get_registry().invalidate()

    resp = client.get("/api/datasets/main--counter/vectors/count")
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["nrows"] == 50
    assert body["ncols"] == 8


# ---------------------------------------------------------------------------
# 3. Dataset not found → 404
# ---------------------------------------------------------------------------


def test_count_dataset_not_found_404(app_client):
    """Non-existent dataset_id returns 404."""
    client, _ = app_client
    resp = client.get("/api/datasets/main--doesnotexist/vectors/count")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4. nrows increments after append
# ---------------------------------------------------------------------------


def test_count_after_append(app_client):
    """After appending M rows, nrows increases by M."""
    client, _ = app_client

    before = client.get("/api/datasets/main--counter/vectors/count").json()
    assert before["nrows"] == 50

    vectors = [[float(i)] * 8 for i in range(10)]
    append_resp = client.post(
        "/api/datasets/main--counter/vectors/append",
        json={"vectors": vectors},
    )
    assert append_resp.status_code == 200, append_resp.json()

    after = client.get("/api/datasets/main--counter/vectors/count").json()
    assert after["nrows"] == 60
    assert after["ncols"] == 8


# ---------------------------------------------------------------------------
# 5. nrows unchanged after overwrite
# ---------------------------------------------------------------------------


def test_count_after_overwrite(app_client):
    """After overwriting K rows, nrows stays the same (shape unchanged)."""
    client, _ = app_client

    before = client.get("/api/datasets/main--counter/vectors/count").json()
    assert before["nrows"] == 50

    overwrite_resp = client.post(
        "/api/datasets/main--counter/vectors/overwrite",
        json={
            "updates": [
                {"row_index": 0, "vector": [9.0] * 8},
                {"row_index": 49, "vector": [7.0] * 8},
            ]
        },
    )
    assert overwrite_resp.status_code == 200, overwrite_resp.json()

    after = client.get("/api/datasets/main--counter/vectors/count").json()
    assert after["nrows"] == 50
    assert after["ncols"] == 8


# ---------------------------------------------------------------------------
# 6. dataset_id echoed correctly
# ---------------------------------------------------------------------------


def test_count_shape_echoed(app_client):
    """Response dataset_id matches the requested path parameter."""
    client, _ = app_client
    resp = client.get("/api/datasets/main--counter/vectors/count")
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == "main--counter"


# ---------------------------------------------------------------------------
# 7. 1-D array → 422 (defensive guard)
# ---------------------------------------------------------------------------


def test_count_1d_array_422(app_client):
    """A 1-D Zarr array registered in the backend raises 422, not 500."""

    from arro_server.storage.base import DatasetSummary
    from arro_server.storage.registry import get_registry

    client, _ = app_client

    # Manually inject a 1-D DatasetSummary into the warm cache.
    # This simulates a corrupted or non-standard dataset that passed
    # upload/commit but is not a 2-D array.
    registry = get_registry()
    fake_summary = DatasetSummary(
        dataset_id="main--flat",
        root="main",
        path="flat",
        shape=(100,),      # 1-D
        dtype="float32",
    )
    with registry._lock:
        registry._ensure_loaded()
        registry._cache["main--flat"] = fake_summary  # type: ignore[index]

    resp = client.get("/api/datasets/main--flat/vectors/count")
    assert resp.status_code == 422
    assert "not 2-d" in resp.json()["detail"].lower()
