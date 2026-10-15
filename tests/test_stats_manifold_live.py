"""tests/test_stats_manifold_live.py

HTTP coverage for GET /api/datasets/{id}/stats and GET /api/datasets/{id}/manifold
on the LIVE-index path (index built via POST /index), complementing the sidecar
fallback coverage in test_api.py and test_phase2_persistence.py.

Test inventory:
    1. test_stats_live_backend           — built index → backend="arrowspace", stats.nitems > 0
    2. test_manifold_live_source         — built index → source="live"
    3. test_stats_404_without_index      — no index → sidecar/none fallback, never 500
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

DATASET_ID = "main--matrix"
STATS_URL = f"/api/datasets/{DATASET_ID}/stats"
MANIFOLD_URL = f"/api/datasets/{DATASET_ID}/manifold"
INDEX_URL = f"/api/datasets/{DATASET_ID}/index"


def _fake_arrowspace_module() -> types.ModuleType:
    """Fake arrowspace module with concrete lambdas_sorted (manifold_data iterates it)."""
    n = 10
    aspace = MagicMock()
    aspace.nitems = n
    aspace.nfeatures = 4
    aspace.nclusters = 2
    aspace.lambdas_sorted.return_value = [
        (float(n - i), i) for i in range(n)
    ]
    gl = MagicMock()
    gl.nnodes = n
    gl.shape = (n, n)
    gl.graph_params = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
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
        yield client

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1-2. Live index path — backend="arrowspace" / source="live"
# ---------------------------------------------------------------------------


def test_stats_live_backend(live_index_client):
    """With a built index, GET /stats returns backend="arrowspace" and real stats."""
    client = live_index_client
    r = client.post(INDEX_URL)
    assert r.status_code == 200, r.text

    resp = client.get(STATS_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend"] == "arrowspace"
    assert body["arrowspace_available"] is True
    assert body["stats"]["nitems"] == 10
    assert body["stats"]["nfeatures"] == 4
    assert body["stats"]["nclusters"] == 2
    assert body["stats"]["gl_nodes"] == 10


def test_manifold_live_source(live_index_client):
    """With a built index, GET /manifold returns source="live" and lambdas_sorted."""
    client = live_index_client
    r = client.post(INDEX_URL)
    assert r.status_code == 200, r.text

    resp = client.get(MANIFOLD_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "live"
    assert body["backend"] == "arrowspace"
    assert body["manifold"]["nitems"] == 10
    assert body["manifold"]["lambdas_sorted"] == [
        [float(10 - i), i] for i in range(10)
    ][:50]


# ---------------------------------------------------------------------------
# 3. No index — fallback path, never 500
# ---------------------------------------------------------------------------


def test_stats_404_without_index(live_index_client):
    """Without a built index, GET /stats falls back to sidecar/none instead of erroring."""
    client = live_index_client
    resp = client.get(STATS_URL)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend"] in {"sidecar", "none"}
