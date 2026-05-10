"""tests/test_phase2_persistence.py — Phase 2 persistence test suite.

Covers:
  1. Manifest read/write/remove (unit)
  2. build_index persists to manifest
  3. load_persisted restores index after cache reset (restart simulation)
  4. delete_index clears cache + manifest + Parquet dir
  5. DELETE /api/datasets/{id}/index HTTP endpoint
  6. GET /api/health includes indexed_datasets
  7. load_persisted is idempotent (double-load)
  8. load_persisted skips broken entries gracefully
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FAKE_ITEMS = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)
FAKE_HITS = [(i, float(i) * 0.01) for i in range(5)]
DATASET_ID = "main--matrix"
VECTOR = [float(i) for i in range(NFEATURES)]


# ---------------------------------------------------------------------------
# Fake arrowspace module with build_and_store + load_arrowspace
# ---------------------------------------------------------------------------


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    aspace.lambdas.return_value = [float(i) * 0.1 for i in range(NITEMS)]
    aspace.lambdas_sorted.return_value = [(float(i) * 0.1, i) for i in range(NITEMS)]
    aspace.get_item.side_effect = lambda idx: FAKE_ITEMS[idx]
    aspace.get_all_items.return_value = FAKE_ITEMS
    aspace.search.return_value = FAKE_HITS
    aspace.search_energy.return_value = FAKE_HITS
    aspace.search_hybrid.return_value = FAKE_HITS
    aspace.search_linear_sorted.return_value = FAKE_HITS
    aspace.search_batch.return_value = [FAKE_HITS]
    return aspace


def _make_fake_gl() -> MagicMock:
    gl = MagicMock()
    gl.nnodes = NITEMS
    gl.shape = (NITEMS, NITEMS)
    gl.graph_params = GRAPH_PARAMS
    return gl


def _make_fake_arrowspace_module(index_store: Path) -> types.ModuleType:
    """Fake module whose build_and_store writes a sentinel file on disk."""
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilder:
        def build_and_store(
            self, graph_params, array, storage_path: str, dataset_name: str
        ):
            # Write a sentinel directory to simulate Parquet output
            dest = Path(storage_path) / dataset_name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "items.parquet").write_text("fake")
            (dest / "graph.parquet").write_text("fake")
            return aspace, gl

    def load_arrowspace(
        storage_path: str,
        dataset_name: str,
        graph_params: dict,
        energy: bool = False,
    ):
        # Verify files exist (simulates real load validation)
        dest = Path(storage_path) / dataset_name
        if not dest.exists():
            raise FileNotFoundError(f"No persisted index at {dest}")
        return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    fake_mod.load_arrowspace = load_arrowspace  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def index_store(tmp_path: Path) -> Path:
    store = tmp_path / "index-store"
    store.mkdir()
    return store


@pytest.fixture
def fake_mod(index_store: Path) -> types.ModuleType:
    return _make_fake_arrowspace_module(index_store)


@pytest.fixture
def adapter(fake_mod, index_store):
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
    return _ArrowSpaceAdapter(fake_mod, cache_size=8)


@pytest.fixture
def live_client(tmp_zarr_root: Path, fake_mod: types.ModuleType, index_store: Path):
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_ARROWSPACE_INDEX_STORE"] = str(index_store)
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()

    app = create_app()
    with TestClient(app) as client:
        yield client

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_ARROWSPACE_INDEX_STORE", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture
def built_client(live_client: TestClient) -> TestClient:
    r = live_client.post(f"/api/datasets/{DATASET_ID}/index")
    assert r.status_code == 200, r.text
    return live_client


# ===========================================================================
# 1. Manifest unit tests
# ===========================================================================


class TestManifest:
    def test_empty_manifest_returns_empty_dict(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        assert m.get_all() == {}

    def test_put_and_get(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        m.put("ds1", "dataset_ds1")
        assert m.get_all() == {"ds1": "dataset_ds1"}

    def test_put_multiple(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        m.put("ds1", "dataset_ds1")
        m.put("ds2", "dataset_ds2")
        assert set(m.get_all().keys()) == {"ds1", "ds2"}

    def test_remove_existing(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        m.put("ds1", "dataset_ds1")
        removed = m.remove("ds1")
        assert removed == "dataset_ds1"
        assert "ds1" not in m.get_all()

    def test_remove_nonexistent_returns_none(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        assert m.remove("never_existed") is None

    def test_put_overwrites(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        m = _Manifest(index_store)
        m.put("ds1", "dataset_ds1_v1")
        m.put("ds1", "dataset_ds1_v2")
        assert m.get_all()["ds1"] == "dataset_ds1_v2"

    def test_manifest_file_created(self, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest, MANIFEST_FILENAME
        m = _Manifest(index_store)
        m.put("ds1", "dataset_ds1")
        assert (index_store / MANIFEST_FILENAME).exists()


# ===========================================================================
# 2. build_index persists to manifest
# ===========================================================================


class TestBuildPersists:
    def test_manifest_written_after_build(self, adapter, index_store: Path):
        from arro_server.arrowspace_adapter import MANIFEST_FILENAME
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        manifest_path = index_store / MANIFEST_FILENAME
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert DATASET_ID in data

    def test_parquet_sentinel_files_written(self, adapter, index_store: Path):
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        dataset_name = adapter._slug(DATASET_ID)
        parquet_dir = index_store / dataset_name
        assert parquet_dir.exists()
        assert (parquet_dir / "items.parquet").exists()
        assert (parquet_dir / "graph.parquet").exists()

    def test_build_returns_dataset_name(self, adapter, index_store: Path):
        meta = adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        assert "dataset_name" in meta
        assert meta["dataset_name"] == adapter._slug(DATASET_ID)

    def test_build_returns_graph_params(self, adapter, index_store: Path):
        meta = adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        assert "graph_params" in meta


# ===========================================================================
# 3. load_persisted — restart simulation
# ===========================================================================


class TestLoadPersisted:
    def test_index_survives_cache_reset(self, adapter, index_store: Path):
        """Build -> clear cache -> load_persisted -> index usable again."""
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        # Simulate server restart: wipe in-memory cache
        adapter._cache.delete(DATASET_ID)
        assert DATASET_ID not in adapter._cache

        n = adapter.load_persisted(index_store)
        assert n == 1
        assert DATASET_ID in adapter._cache

        # Index is functional after reload
        result = adapter.search(DATASET_ID, {"vector": VECTOR, "tau": 1.0})
        assert result["backend"] == "arrowspace"

    def test_load_persisted_empty_store(self, adapter, index_store: Path):
        n = adapter.load_persisted(index_store)
        assert n == 0

    def test_load_persisted_idempotent(self, adapter, index_store: Path):
        """Calling load_persisted twice does not duplicate cache entries."""
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        adapter._cache.delete(DATASET_ID)

        adapter.load_persisted(index_store)
        adapter.load_persisted(index_store)  # second call — already in cache
        # Still exactly one entry
        assert adapter._cache.keys().count(DATASET_ID) == 1

    def test_load_persisted_skips_broken_entry(self, adapter, index_store: Path):
        """A manifest entry whose Parquet dir is missing is skipped gracefully."""
        from arro_server.arrowspace_adapter import _Manifest
        # Write a manifest entry that points to a non-existent directory
        _Manifest(index_store).put("ghost_dataset", "dataset_ghost")
        # Should not raise; ghost entry skipped, returns 0
        n = adapter.load_persisted(index_store)
        assert n == 0
        assert "ghost_dataset" not in adapter._cache

    def test_load_persisted_multiple_datasets(self, adapter, index_store: Path):
        adapter.build_index("main--a", FAKE_ITEMS.copy(), index_store)
        adapter.build_index("main--b", FAKE_ITEMS.copy(), index_store)
        # Clear both
        adapter._cache.delete("main--a")
        adapter._cache.delete("main--b")
        n = adapter.load_persisted(index_store)
        assert n == 2
        assert "main--a" in adapter._cache
        assert "main--b" in adapter._cache


# ===========================================================================
# 4. delete_index
# ===========================================================================


class TestDeleteIndex:
    def test_delete_removes_from_cache(self, adapter, index_store: Path):
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        assert DATASET_ID in adapter._cache
        result = adapter.delete_index(DATASET_ID, index_store)
        assert result is True
        assert DATASET_ID not in adapter._cache

    def test_delete_removes_from_manifest(self, adapter, index_store: Path):
        from arro_server.arrowspace_adapter import _Manifest
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        adapter.delete_index(DATASET_ID, index_store)
        assert DATASET_ID not in _Manifest(index_store).get_all()

    def test_delete_removes_parquet_dir(self, adapter, index_store: Path):
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        dataset_name = adapter._slug(DATASET_ID)
        parquet_dir = index_store / dataset_name
        assert parquet_dir.exists()
        adapter.delete_index(DATASET_ID, index_store)
        assert not parquet_dir.exists()

    def test_delete_nonexistent_returns_false(self, adapter, index_store: Path):
        assert adapter.delete_index("never_built", index_store) is False

    def test_delete_then_rebuild_works(self, adapter, index_store: Path):
        adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        adapter.delete_index(DATASET_ID, index_store)
        meta = adapter.build_index(DATASET_ID, FAKE_ITEMS.copy(), index_store)
        assert meta["nitems"] == NITEMS
        assert DATASET_ID in adapter._cache


# ===========================================================================
# 5. DELETE /api/datasets/{id}/index — HTTP
# ===========================================================================


class TestDeleteIndexHTTP:
    def test_delete_index_200(self, built_client: TestClient):
        r = built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] is True
        assert body["id"] == DATASET_ID

    def test_delete_index_404_when_not_built(self, live_client: TestClient):
        r = live_client.delete(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 404

    def test_delete_index_unknown_dataset_404(self, built_client: TestClient):
        r = built_client.delete("/api/datasets/main--missing/index")
        assert r.status_code == 404

    def test_search_after_delete_returns_error(self, built_client: TestClient):
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        r = built_client.post(
            f"/api/datasets/{DATASET_ID}/search",
            json={"vector": VECTOR},
        )
        assert r.status_code in {404, 503}

    def test_rebuild_after_delete_works(self, built_client: TestClient):
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        r = built_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        assert r.json()["built"] is True


# ===========================================================================
# 6. GET /api/health — indexed_datasets
# ===========================================================================


class TestHealthIndexedDatasets:
    def test_health_has_indexed_datasets_field(self, live_client: TestClient):
        r = live_client.get("/api/health")
        assert r.status_code == 200
        assert "indexed_datasets" in r.json()

    def test_health_empty_before_build(self, live_client: TestClient):
        body = live_client.get("/api/health").json()
        assert body["indexed_datasets"] == []

    def test_health_lists_dataset_after_build(self, built_client: TestClient):
        body = built_client.get("/api/health").json()
        assert DATASET_ID in body["indexed_datasets"]

    def test_health_removes_dataset_after_delete(self, built_client: TestClient):
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        body = built_client.get("/api/health").json()
        assert DATASET_ID not in body["indexed_datasets"]
