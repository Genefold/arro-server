"""tests/test_vectors_stale_signal.py

Tests for the index_stale signal on POST .../vectors/append and
POST .../vectors/overwrite (issue #56 / F-003).

Test inventory:
    1. test_append_reports_index_stale_true       — append with index → True
    2. test_append_reports_index_stale_false_no_index — append, no index → False
    3. test_overwrite_reports_index_stale_true    — overwrite with index → True
    4. test_overwrite_reports_index_stale_false_no_index — overwrite, no index → False
    5. test_append_index_stale_cleared_after_rebuild — rebuild via POST /index, append → True again
    6. test_index_stale_false_after_index_deleted — DELETE /index, append → False

Tests 1-4 use a mocked adapter (pattern from test_upload.py).
Tests 5-6 use the fake arrowspace module (pattern from test_phase2_persistence.py)
so the real adapter + real POST/DELETE /index routes are exercised.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

DATASET_ID = "main--matrix"
APPEND_URL = f"/api/datasets/{DATASET_ID}/vectors/append"
OVERWRITE_URL = f"/api/datasets/{DATASET_ID}/vectors/overwrite"
INDEX_URL = f"/api/datasets/{DATASET_ID}/index"

APPEND_BODY = {"vectors": [[float(i) for i in range(4)] for _ in range(2)], "dtype": "float32"}
OVERWRITE_BODY = {"updates": [{"row_index": 0, "vector": [9.0, 9.0, 9.0, 9.0]}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client(tmp_path: Path):
    """TestClient with a single 2-D Zarr array at main--matrix (50, 4) float32."""
    import zarr

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    zarr_dir = tmp_path / "matrix"
    arr = zarr.open(str(zarr_dir), mode="w", shape=(50, 4), chunks=(10, 4), dtype="float32")
    arr[:] = np.arange(50 * 4, dtype="float32").reshape(50, 4)

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


def _fake_arrowspace_module() -> types.ModuleType:
    """Fake arrowspace module: ArrowSpaceBuilder returns stub aspace/gl."""
    n = 10
    aspace = MagicMock()
    aspace.nitems = n
    aspace.nfeatures = 4
    aspace.nclusters = 2
    gl = MagicMock()
    gl.nnodes = n
    gl.shape = (n, n)
    gl.to_csr.return_value = (
        np.ones(n, dtype=np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64),
        (n, n),
    )

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod = types.ModuleType("arrowspace")
    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    return fake_mod


@pytest.fixture()
def live_index_client(tmp_path: Path):
    """TestClient with a real (fake-module-backed) adapter and isolated index store."""
    import zarr

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    fake_mod = _fake_arrowspace_module()

    root_dir = tmp_path / "data"
    root_dir.mkdir()
    zarr_dir = root_dir / "matrix"
    arr = zarr.open(
        str(zarr_dir), mode="w", shape=(50, 4), chunks=(10, 4), dtype="float32"
    )
    arr[:] = np.arange(50 * 4, dtype="float32").reshape(50, 4)

    index_store = tmp_path / "index_store"
    index_store.mkdir()

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]
    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={root_dir}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(index_store)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, index_store

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1-4. Mocked adapter — signal reflects has_index() result
# ---------------------------------------------------------------------------


def test_append_reports_index_stale_true(app_client):
    """Append on a dataset that has an index returns index_stale=True."""
    client, _ = app_client
    mock_adapter = MagicMock()
    mock_adapter.has_index.return_value = True
    with patch("arro_server.api.routes.load_arrowspace", return_value=mock_adapter):
        resp = client.post(APPEND_URL, json=APPEND_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is True
    mock_adapter.has_index.assert_called_once_with(DATASET_ID)


def test_append_reports_index_stale_false_no_index(app_client):
    """Append on a dataset without an index returns index_stale=False."""
    client, _ = app_client
    mock_adapter = MagicMock()
    mock_adapter.has_index.return_value = False
    with patch("arro_server.api.routes.load_arrowspace", return_value=mock_adapter):
        resp = client.post(APPEND_URL, json=APPEND_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is False


def test_overwrite_reports_index_stale_true(app_client):
    """Overwrite on a dataset that has an index returns index_stale=True."""
    client, _ = app_client
    mock_adapter = MagicMock()
    mock_adapter.has_index.return_value = True
    with patch("arro_server.api.routes.load_arrowspace", return_value=mock_adapter):
        resp = client.post(OVERWRITE_URL, json=OVERWRITE_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is True
    mock_adapter.has_index.assert_called_once_with(DATASET_ID)


def test_overwrite_reports_index_stale_false_no_index(app_client):
    """Overwrite on a dataset without an index returns index_stale=False."""
    client, _ = app_client
    mock_adapter = MagicMock()
    mock_adapter.has_index.return_value = False
    with patch("arro_server.api.routes.load_arrowspace", return_value=mock_adapter):
        resp = client.post(OVERWRITE_URL, json=OVERWRITE_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is False


# ---------------------------------------------------------------------------
# 5-6. Real adapter — signal not sticky, tracks index lifecycle
# ---------------------------------------------------------------------------


def test_append_index_stale_cleared_after_rebuild(live_index_client):
    """Rebuild via POST /index, then append → index_stale=True again (not a latch)."""
    client, _ = live_index_client

    r = client.post(INDEX_URL)
    assert r.status_code == 200, r.text

    resp = client.post(APPEND_URL, json=APPEND_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is True

    # Rebuild the index (which the first append made stale).
    r = client.post(INDEX_URL)
    assert r.status_code == 200, r.text

    resp = client.post(APPEND_URL, json=APPEND_BODY)
    assert resp.status_code == 200, resp.json()
    # The rebuild created a fresh index which this append now makes stale again.
    assert resp.json()["index_stale"] is True


def test_index_stale_false_after_index_deleted(live_index_client):
    """DELETE /index, then append → index_stale=False (has_index is not sticky)."""
    client, _ = live_index_client

    r = client.post(INDEX_URL)
    assert r.status_code == 200, r.text

    r = client.delete(INDEX_URL)
    assert r.status_code == 200, r.text

    resp = client.post(APPEND_URL, json=APPEND_BODY)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["index_stale"] is False
