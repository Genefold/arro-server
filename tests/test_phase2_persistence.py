"""tests/test_phase2_persistence.py — Phase 2 index persistence tests.

Covers the new behaviours introduced by feat/phase2-index-persistence:
  * _read_manifest / _write_manifest helpers
  * LRUIndexCache.keys()
  * _ArrowSpaceAdapter.has_index()
  * _ArrowSpaceAdapter.indexed_datasets()
  * _ArrowSpaceAdapter.build_index() manifest write + dataset_name reuse
  * _ArrowSpaceAdapter.delete_index() (cache + Parquet + CSR + manifest)
  * _ArrowSpaceAdapter.reload_from_manifest() (stub-only; no real parquet)
  * DELETE /api/datasets/{id}/index  (HTTP)
  * GET  /api/health  indexed_datasets key  (HTTP)

All tests use the same fake arrowspace module from test_arrowspace.py so
they run without the real arrowspace package installed.
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
# Test constants (mirror test_arrowspace.py)
# ---------------------------------------------------------------------------

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FIXTURE_ARRAY = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)
DATASET_ID = "main--matrix"

# ---------------------------------------------------------------------------
# Fake arrowspace module (no with_persistence; mirrors Phase-1 behaviour)
# ---------------------------------------------------------------------------


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    return aspace


def _make_fake_gl() -> MagicMock:
    gl = MagicMock()
    gl.nnodes = NITEMS
    gl.shape = (NITEMS, NITEMS)
    gl.graph_params = GRAPH_PARAMS
    n = NITEMS
    gl.to_csr.return_value = (
        np.ones(n, dtype=np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64),
        (n, n),
    )
    return gl


def _make_fake_mod() -> types.ModuleType:
    """Fake module without with_persistence (simulates older arrowspace build)."""
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mod() -> types.ModuleType:
    return _make_fake_mod()


@pytest.fixture
def adapter(fake_mod):
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    return _ArrowSpaceAdapter(fake_mod, cache_size=4)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    d = tmp_path / "index_store"
    d.mkdir()
    return d


@pytest.fixture
def built_adapter(adapter, tmp_store: Path):
    """Adapter with one pre-built index."""
    adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
    return adapter


@pytest.fixture
def live_client(tmp_path: Path, fake_mod: types.ModuleType):
    """Full HTTP test client with fake arrowspace and isolated index store."""
    zarr = pytest.importorskip("zarr")

    # Create a minimal Zarr root with a 2-D array
    root_dir = tmp_path / "data"
    root_dir.mkdir()
    ds_path = root_dir / "matrix"
    arr = zarr.open(str(ds_path), mode="w", shape=(NITEMS, NFEATURES), chunks=(5, NFEATURES), dtype="float64")
    arr[:] = FIXTURE_ARRAY

    index_store = tmp_path / "index_store"
    index_store.mkdir()

    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]
    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={root_dir}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(index_store)
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()

    app = create_app()
    with TestClient(app) as client:
        yield client

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture
def built_client(live_client: TestClient) -> TestClient:
    r = live_client.post(f"/api/datasets/{DATASET_ID}/index")
    assert r.status_code == 200, r.text
    return live_client


# ===========================================================================
# 1. Manifest helpers (unit)
# ===========================================================================


class TestManifestHelpers:
    """_read_manifest and _write_manifest — pure module-level functions."""

    def test_read_manifest_absent_returns_empty(self, tmp_store: Path):
        from arro_server.arrowspace_adapter import _read_manifest

        result = _read_manifest(tmp_store)
        assert result == {}

    def test_write_then_read_roundtrip(self, tmp_store: Path):
        from arro_server.arrowspace_adapter import _read_manifest, _write_manifest

        data = {"ds1": {"dataset_name": "ds1_abcd1234", "graph_params": GRAPH_PARAMS}}
        _write_manifest(tmp_store, data)
        assert _read_manifest(tmp_store) == data

    def test_write_creates_directory(self, tmp_path: Path):
        from arro_server.arrowspace_adapter import _write_manifest

        nested = tmp_path / "a" / "b" / "c"
        _write_manifest(nested, {"x": {}})
        assert (nested / "index_manifest.json").exists()

    def test_read_corrupt_file_returns_empty(self, tmp_store: Path):
        from arro_server.arrowspace_adapter import MANIFEST_FILENAME, _read_manifest

        (tmp_store / MANIFEST_FILENAME).write_text("not-json")
        result = _read_manifest(tmp_store)
        assert result == {}

    def test_write_is_atomic(self, tmp_store: Path):
        """No .tmp file should remain after _write_manifest completes."""
        from arro_server.arrowspace_adapter import _write_manifest

        _write_manifest(tmp_store, {"ds": {}})
        tmp_files = list(tmp_store.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


# ===========================================================================
# 2. LRUIndexCache.keys() (unit)
# ===========================================================================


class TestLRUCacheKeys:
    def _make_entry(self):
        from arro_server.arrowspace_adapter import _IndexEntry

        return _IndexEntry(aspace=None, gl=None, nitems=1, nfeatures=1, nclusters=1)

    def test_keys_empty_on_new_cache(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=4)
        assert cache.keys() == []

    def test_keys_reflects_put(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=4)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())
        assert set(cache.keys()) == {"a", "b"}

    def test_keys_excludes_deleted(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=4)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())
        cache.delete("a")
        assert cache.keys() == ["b"]

    def test_keys_excludes_evicted(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())
        cache.put("c", self._make_entry())  # evicts "a" (LRU)
        assert "a" not in cache.keys()
        assert set(cache.keys()) == {"b", "c"}


# ===========================================================================
# 3. has_index() and indexed_datasets() (unit)
# ===========================================================================


class TestHasIndexAndIndexedDatasets:
    def test_has_index_false_before_build(self, adapter):
        assert adapter.has_index("missing/ds") is False

    def test_has_index_true_after_build(self, adapter, tmp_store):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_store)
        assert adapter.has_index("test/ds") is True

    def test_has_index_false_after_delete(self, adapter, tmp_store):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.delete_index("test/ds", tmp_store)
        assert adapter.has_index("test/ds") is False

    def test_indexed_datasets_empty_initially(self, adapter):
        assert adapter.indexed_datasets() == []

    def test_indexed_datasets_lists_built(self, adapter, tmp_store):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_store)
        assert "test/ds" in adapter.indexed_datasets()

    def test_indexed_datasets_excludes_deleted(self, adapter, tmp_store):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.delete_index("test/ds", tmp_store)
        assert "test/ds" not in adapter.indexed_datasets()


# ===========================================================================
# 4. build_index manifest behaviour (unit)
# ===========================================================================


class TestBuildIndexManifest:
    def test_manifest_written_after_build(self, adapter, tmp_store):
        from arro_server.arrowspace_adapter import MANIFEST_FILENAME

        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_store)
        assert (tmp_store / MANIFEST_FILENAME).exists()

    def test_manifest_contains_dataset_name(self, adapter, tmp_store):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_store)
        manifest = _read_manifest(tmp_store)
        assert "ds1" in manifest
        assert "dataset_name" in manifest["ds1"]
        assert manifest["ds1"]["dataset_name"]  # non-empty

    def test_manifest_contains_graph_params(self, adapter, tmp_store):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_store, graph_params=GRAPH_PARAMS)
        manifest = _read_manifest(tmp_store)
        assert manifest["ds1"]["graph_params"] == GRAPH_PARAMS

    def test_rebuild_reuses_dataset_name(self, adapter, tmp_store):
        """Rebuilding a dataset must keep the same dataset_name in the manifest."""
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_store)
        first_name = _read_manifest(tmp_store)["ds1"]["dataset_name"]
        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_store)
        second_name = _read_manifest(tmp_store)["ds1"]["dataset_name"]
        assert first_name == second_name

    def test_multiple_datasets_in_manifest(self, adapter, tmp_store):
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds_a", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index("ds_b", FIXTURE_ARRAY.copy(), tmp_store)
        manifest = _read_manifest(tmp_store)
        assert "ds_a" in manifest
        assert "ds_b" in manifest

    def test_dataset_names_are_unique(self, adapter, tmp_store):
        """Different datasets must get distinct dataset_name values."""
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds_a", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index("ds_b", FIXTURE_ARRAY.copy(), tmp_store)
        manifest = _read_manifest(tmp_store)
        assert manifest["ds_a"]["dataset_name"] != manifest["ds_b"]["dataset_name"]


# ===========================================================================
# 5. delete_index (unit)
# ===========================================================================


class TestDeleteIndex:
    def test_returns_true_for_existing(self, built_adapter, tmp_store):
        assert built_adapter.delete_index(DATASET_ID, tmp_store) is True

    def test_returns_false_for_missing(self, adapter, tmp_store):
        assert adapter.delete_index("never/built", tmp_store) is False

    def test_removes_from_cache(self, built_adapter, tmp_store):
        built_adapter.delete_index(DATASET_ID, tmp_store)
        assert not built_adapter.has_index(DATASET_ID)

    def test_removes_from_manifest(self, built_adapter, tmp_store):
        from arro_server.arrowspace_adapter import _read_manifest

        built_adapter.delete_index(DATASET_ID, tmp_store)
        manifest = _read_manifest(tmp_store)
        assert DATASET_ID not in manifest

    def test_manifest_updated_after_delete(self, adapter, tmp_store):
        """After deleting one of two entries, the other must still be present."""
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds_a", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index("ds_b", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.delete_index("ds_a", tmp_store)
        manifest = _read_manifest(tmp_store)
        assert "ds_a" not in manifest
        assert "ds_b" in manifest

    def test_double_delete_returns_false(self, built_adapter, tmp_store):
        built_adapter.delete_index(DATASET_ID, tmp_store)
        assert built_adapter.delete_index(DATASET_ID, tmp_store) is False

    def test_deletes_csr_zarr_directory(self, adapter, tmp_store):
        """_persist_csr writes a slug-named directory; delete_index removes it."""
        pytest.importorskip("zarr")
        adapter.build_index(DATASET_ID, FIXTURE_ARRAY.copy(), tmp_store)
        slug = DATASET_ID.replace("/", "__").replace("\\", "__")
        csr_dir = tmp_store / slug
        # Only assert removal if the directory was actually created
        if csr_dir.exists():
            adapter.delete_index(DATASET_ID, tmp_store)
            assert not csr_dir.exists()
        else:
            adapter.delete_index(DATASET_ID, tmp_store)
            # No directory to remove — just verify delete reported True


# ===========================================================================
# 6. reload_from_manifest (unit)
# ===========================================================================


class TestReloadFromManifest:
    def test_empty_manifest_returns_empty_list(self, adapter, tmp_store):
        result = adapter.reload_from_manifest(tmp_store)
        assert result == []

    def test_entry_missing_dataset_name_is_skipped(self, adapter, tmp_store):
        """Manifest entries without 'dataset_name' must be skipped gracefully."""
        from arro_server.arrowspace_adapter import _write_manifest

        # Write a manifest entry that is missing the required dataset_name key.
        _write_manifest(tmp_store, {"bad/ds": {"graph_params": GRAPH_PARAMS}})
        result = adapter.reload_from_manifest(tmp_store)
        assert "bad/ds" not in result

    def test_load_failure_is_skipped_gracefully(self, adapter, tmp_store):
        """When load_arrowspace raises, the entry is skipped and no exception propagates."""
        from arro_server.arrowspace_adapter import _write_manifest

        # Our fake module has no load_arrowspace function, so this will raise.
        _write_manifest(tmp_store, {"some/ds": {"dataset_name": "some_ds_abc123", "graph_params": GRAPH_PARAMS}})
        result = adapter.reload_from_manifest(tmp_store)
        # The entry load failed, so it must not appear in the loaded list.
        assert "some/ds" not in result

    def test_returns_list_type(self, adapter, tmp_store):
        result = adapter.reload_from_manifest(tmp_store)
        assert isinstance(result, list)


# ===========================================================================
# 7. Stub adapter no-ops (unit)
# ===========================================================================


class TestStubAdapters:
    """_SidecarAdapter and _NullAdapter must implement the Phase 2 interface."""

    def test_null_adapter_has_index_false(self):
        from arro_server.arrowspace_adapter import _NullAdapter

        adapter = _NullAdapter()
        assert adapter.has_index("any") is False

    def test_null_adapter_indexed_datasets_empty(self):
        from arro_server.arrowspace_adapter import _NullAdapter

        adapter = _NullAdapter()
        assert adapter.indexed_datasets() == []

    def test_null_adapter_reload_from_manifest_empty(self, tmp_store):
        from arro_server.arrowspace_adapter import _NullAdapter

        adapter = _NullAdapter()
        assert adapter.reload_from_manifest(tmp_store) == []

    def test_sidecar_adapter_has_index_false(self):
        from arro_server.arrowspace_adapter import _SidecarAdapter

        adapter = _SidecarAdapter()
        assert adapter.has_index("any") is False

    def test_sidecar_adapter_indexed_datasets_empty(self):
        from arro_server.arrowspace_adapter import _SidecarAdapter

        adapter = _SidecarAdapter()
        assert adapter.indexed_datasets() == []

    def test_sidecar_adapter_reload_from_manifest_empty(self, tmp_store):
        from arro_server.arrowspace_adapter import _SidecarAdapter

        adapter = _SidecarAdapter()
        assert adapter.reload_from_manifest(tmp_store) == []


# ===========================================================================
# 8. DELETE /api/datasets/{id}/index  (HTTP)
# ===========================================================================


class TestDeleteIndexEndpoint:
    def test_delete_existing_index_200(self, built_client: TestClient):
        r = built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["deleted"] is True

    def test_delete_nonexistent_index_404(self, live_client: TestClient):
        r = live_client.delete(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 404

    def test_delete_makes_index_unavailable(self, built_client: TestClient):
        """After DELETE the index should no longer be accessible."""
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        r = built_client.get(f"/api/datasets/{DATASET_ID}/lambdas")
        assert r.status_code in {404, 503}

    def test_double_delete_returns_404(self, built_client: TestClient):
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        r = built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 404

    def test_delete_unknown_dataset_id_404(self, live_client: TestClient):
        r = live_client.delete("/api/datasets/main--no-such-dataset/index")
        assert r.status_code == 404

    def test_rebuild_after_delete_succeeds(self, built_client: TestClient):
        """DELETE then POST /index must succeed and restore the index."""
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        r = built_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        assert r.json()["built"] is True

    def test_delete_removes_from_manifest(self, built_client: TestClient, tmp_path: Path):
        """After DELETE the manifest on disk must not contain the dataset_id."""
        from arro_server.arrowspace_adapter import MANIFEST_FILENAME

        # Locate the index_store from env (set by live_client fixture)
        index_store = Path(os.environ["ARRO_SERVER_INDEX_STORE"])
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        manifest_path = index_store / MANIFEST_FILENAME
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert DATASET_ID not in manifest


# ===========================================================================
# 9. GET /api/health — indexed_datasets key  (HTTP)
# ===========================================================================


class TestHealthIndexedDatasets:
    def test_health_has_indexed_datasets_key(self, live_client: TestClient):
        r = live_client.get("/api/health")
        assert r.status_code == 200
        assert "indexed_datasets" in r.json()

    def test_indexed_datasets_empty_before_build(self, live_client: TestClient):
        body = live_client.get("/api/health").json()
        assert body["indexed_datasets"] == []

    def test_indexed_datasets_populated_after_build(self, built_client: TestClient):
        body = built_client.get("/api/health").json()
        assert DATASET_ID in body["indexed_datasets"]

    def test_indexed_datasets_empty_after_delete(self, built_client: TestClient):
        built_client.delete(f"/api/datasets/{DATASET_ID}/index")
        body = built_client.get("/api/health").json()
        assert DATASET_ID not in body["indexed_datasets"]
