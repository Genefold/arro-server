"""tests/test_arrowspace.py — Consolidated ArrowSpace test suite.

Merges test_phase1_arrowspace.py and test_arrowspace_adapter.py into one
canonical module.

Design decisions
----------------
* All tests use the fake ArrowSpaceBuilder (no pytest.importorskip) so the
  full suite runs in any CI environment regardless of whether the real
  ``arrowspace`` package is installed.
* Basic, deterministic test data: NITEMS=10, NFEATURES=4 throughout.
* Adapter is fully implemented — build_index writes zzarr files under the
  supplied index_path and caches the built objects in the LRU.
* New zzarr-persistence assertions verify that the on-disk ArrowSpace index
  files (data.zarr, indices.zarr, indptr.zarr, meta.json) are created both
  via the adapter directly and via the HTTP POST /index endpoint.

Implementation notes
--------------------
* _persist_csr writes {nitems, nfeatures, nclusters, csr_shape} to meta.json.
  It does NOT write graph_params — tests assert on csr_shape accordingly.
* The route handler passes settings.index_store (ARRO_SERVER_INDEX_STORE)
  as the index_store argument, NOT the zarr data root.  The live_client
  fixture therefore sets ARRO_SERVER_INDEX_STORE to a dedicated tmp dir so
  HTTP-level zzarr tests can locate the artifacts.
* _ArrowSpaceAdapter._slug replaces '/' and '\\' with '__' but leaves '-'
  untouched: DATASET_ID='main--matrix' -> slug 'main--matrix'.
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
# Basic test data
# ---------------------------------------------------------------------------

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}

# Deterministic 10x4 float64 array -- used everywhere
FIXTURE_ARRAY = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)

FAKE_LAMBDAS = [float(i) * 0.1 for i in range(NITEMS)]
FAKE_HITS = [(i, float(i) * 0.01) for i in range(5)]

DATASET_ID = "main--matrix"
VECTOR = FIXTURE_ARRAY[0].tolist()  # [0.0, 1.0, 2.0, 3.0]

# Slug produced by _ArrowSpaceAdapter._slug(DATASET_ID):
# only '/' and '\\' are replaced by '__'; '-' is left untouched.
DATASET_SLUG = DATASET_ID.replace("/", "__").replace("\\", "__")  # "main--matrix"

# ---------------------------------------------------------------------------
# Fake arrowspace module
# ---------------------------------------------------------------------------


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    aspace.lambdas.return_value = FAKE_LAMBDAS
    aspace.lambdas_sorted.return_value = [(float(v), i) for i, v in enumerate(FAKE_LAMBDAS)]
    aspace.get_item.side_effect = lambda idx: FIXTURE_ARRAY[idx]
    aspace.get_all_items.return_value = FIXTURE_ARRAY
    aspace.search.return_value = FAKE_HITS
    aspace.search_batch.return_value = [FAKE_HITS, FAKE_HITS]
    aspace.search_energy.return_value = FAKE_HITS
    aspace.search_hybrid.return_value = FAKE_HITS
    aspace.search_linear_sorted.return_value = FAKE_HITS
    aspace.spot_motives_eigen.return_value = FAKE_HITS
    aspace.spot_motives_energy.return_value = FAKE_HITS
    aspace.spot_subg_centroids.return_value = FAKE_HITS
    aspace.spot_subg_motives.return_value = FAKE_HITS
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
    gl.to_dense.return_value = np.eye(n, dtype=np.float32)
    return gl


def _make_fake_arrowspace_module() -> types.ModuleType:
    """Return a types.ModuleType that mimics the real arrowspace package."""
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    return fake_mod


def _make_tracking_arrowspace_module() -> types.ModuleType:
    """Return a fake arrowspace module that counts build() calls.

    Used to verify that ``build_index`` calls the builder exactly once.
    """
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class TrackingBuilder:
        build_count: int = 0

        def build(self, graph_params, array):
            TrackingBuilder.build_count += 1
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = TrackingBuilder  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# Shared adapter fixture (unit tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mod() -> types.ModuleType:
    return _make_fake_arrowspace_module()


@pytest.fixture
def adapter(fake_mod):
    """A fresh _ArrowSpaceAdapter backed by the fake arrowspace module."""
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    return _ArrowSpaceAdapter(fake_mod, cache_size=4)


@pytest.fixture
def built_adapter(adapter, tmp_path: Path):
    """An adapter with one pre-built index ('test/ds')."""
    adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
    return adapter


@pytest.fixture
def tracking_adapter(tmp_path: Path):
    """An adapter backed by the tracking builder module."""
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    track_mod = _make_tracking_arrowspace_module()
    return _ArrowSpaceAdapter(track_mod, cache_size=4), track_mod


# ---------------------------------------------------------------------------
# HTTP client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_index_store(tmp_path: Path) -> Path:
    """Isolated directory used as ARRO_SERVER_INDEX_STORE for HTTP tests."""
    d = tmp_path / "index_store"
    d.mkdir()
    return d


@pytest.fixture
def live_client(tmp_zarr_root: Path, tmp_index_store: Path, fake_mod: types.ModuleType):
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(tmp_index_store)
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
# 1. Adapter unit tests
# ===========================================================================


class TestAdapterBuildIndex:
    """build_index builds, caches, and persists the index."""

    def test_returns_expected_keys(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert set(meta) >= {"nitems", "nfeatures", "nclusters"}

    def test_nitems_matches_array(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert meta["nitems"] == NITEMS

    def test_nfeatures_matches_array(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert meta["nfeatures"] == NFEATURES

    def test_nclusters_present(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert meta["nclusters"] == NCLUSTERS

    def test_entry_cached(self, adapter, tmp_path):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert adapter._cache.get("test/ds") is not None

    def test_rejects_1d_array(self, adapter, tmp_path):
        with pytest.raises(ValueError, match="2-D"):
            adapter.build_index("test/ds", np.ones(10), tmp_path)

    def test_custom_graph_params(self, adapter, tmp_path):
        custom = {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path, graph_params=custom)
        assert meta["nitems"] == NITEMS

    def test_rebuild_replaces_cache(self, adapter, tmp_path):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        meta2 = adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert meta2["nitems"] == NITEMS

    # -----------------------------------------------------------------------
    # zzarr persistence assertions
    # -----------------------------------------------------------------------

    def test_zzarr_files_created(self, adapter, tmp_path):
        """build_index must write data.zarr, indices.zarr, indptr.zarr, meta.json."""
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        slug_dir = tmp_path / "test__ds"
        assert (slug_dir / "data.zarr").exists(), "data.zarr missing"
        assert (slug_dir / "indices.zarr").exists(), "indices.zarr missing"
        assert (slug_dir / "indptr.zarr").exists(), "indptr.zarr missing"
        assert (slug_dir / "meta.json").exists(), "meta.json missing"

    def test_meta_json_content(self, adapter, tmp_path):
        """meta.json must record nitems, nfeatures, nclusters and csr_shape.

        Note: _persist_csr writes {nitems, nfeatures, nclusters, csr_shape}.
        graph_params is NOT persisted to disk — it is returned in the HTTP
        response body by the route handler from the in-memory graph_params dict.
        """
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        slug_dir = tmp_path / "test__ds"
        meta = json.loads((slug_dir / "meta.json").read_text())
        assert meta["nitems"] == NITEMS
        assert meta["nfeatures"] == NFEATURES
        assert "csr_shape" in meta, f"expected csr_shape in meta.json, got keys: {list(meta)}"
        assert meta["csr_shape"] == [NITEMS, NITEMS]


class TestAdapterSingleBuild:
    """Verifies that build_index calls the builder exactly once (no double build)."""

    def test_build_called_once(self, tracking_adapter, tmp_path):
        adapter, track_mod = tracking_adapter
        stub = track_mod.ArrowSpaceBuilder
        stub.build_count = 0
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert stub.build_count == 1, (
            f"Expected 1 build() call, got {stub.build_count} — "
            "build_index must not call the builder more than once"
        )

    def test_multiple_builds_each_call_once(self, tracking_adapter, tmp_path):
        adapter, track_mod = tracking_adapter
        stub = track_mod.ArrowSpaceBuilder
        stub.build_count = 0
        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), tmp_path)
        assert stub.build_count == 1
        adapter.build_index("ds2", FIXTURE_ARRAY.copy(), tmp_path)
        assert stub.build_count == 2

    def test_cache_populated_after_build(self, adapter, tmp_path):
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
        assert adapter.has_index("test/ds")
        info = adapter.graph_laplacian_info("test/ds")
        assert info["nnodes"] == NITEMS


class TestAdapterLambdas:
    """lambdas() returns valid eigenvalue data."""

    def test_returns_expected_keys(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert set(result) >= {"nitems", "lambdas", "lambdas_sorted"}

    def test_lambdas_is_list_of_floats(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert isinstance(result["lambdas"], list)
        assert all(isinstance(v, float) for v in result["lambdas"])

    def test_lambdas_sorted_structure(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        for pair in result["lambdas_sorted"]:
            assert len(pair) == 2

    def test_lambdas_length(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert len(result["lambdas"]) == NITEMS

    def test_raises_if_no_index(self, adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("nonexistent/ds")


class TestAdapterGraphLaplacian:
    """graph_laplacian_info() returns shape/params metadata."""

    def test_nnodes(self, built_adapter):
        info = built_adapter.graph_laplacian_info("test/ds")
        assert info["nnodes"] == NITEMS

    def test_shape(self, built_adapter):
        info = built_adapter.graph_laplacian_info("test/ds")
        assert info["shape"] == [NITEMS, NITEMS]

    def test_graph_params(self, built_adapter):
        info = built_adapter.graph_laplacian_info("test/ds")
        assert info["graph_params"] == GRAPH_PARAMS


class TestAdapterItems:
    """get_item / get_all_items correctness."""

    def test_get_item_index(self, built_adapter):
        result = built_adapter.get_item("test/ds", 0)
        assert result["item_index"] == 0

    def test_get_item_vector_length(self, built_adapter):
        result = built_adapter.get_item("test/ds", 0)
        assert len(result["vector"]) == NFEATURES

    def test_get_item_vector_values(self, built_adapter):
        result = built_adapter.get_item("test/ds", 0)
        assert result["vector"] == [float(v) for v in FIXTURE_ARRAY[0]]

    def test_get_all_items_nitems(self, built_adapter):
        result = built_adapter.get_all_items("test/ds")
        assert result["nitems"] == NITEMS

    def test_get_all_items_length(self, built_adapter):
        result = built_adapter.get_all_items("test/ds")
        assert len(result["items"]) == NITEMS
        assert len(result["items"][0]) == NFEATURES


class TestAdapterSearch:
    """search / search_batch / search_energy / search_hybrid / search_linear_sorted."""

    def test_search_returns_results(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": VECTOR, "tau": 1.0})
        assert result["backend"] == "arrowspace"
        assert len(result["results"]) == len(FAKE_HITS)
        assert "index" in result["results"][0]
        assert "score" in result["results"][0]

    def test_search_requires_vector(self, built_adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            built_adapter.search("test/ds", {"tau": 1.0})

    def test_search_custom_tau(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": VECTOR, "tau": 2.0})
        assert "results" in result

    def test_search_batch(self, built_adapter):
        result = built_adapter.search_batch("test/ds", {"vectors": [VECTOR, VECTOR], "tau": 1.0})
        assert len(result["results"]) == 2
        assert len(result["results"][0]) == len(FAKE_HITS)

    def test_search_batch_requires_vectors(self, built_adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            built_adapter.search_batch("test/ds", {"tau": 1.0})

    def test_search_energy(self, built_adapter):
        result = built_adapter.search_energy("test/ds", {"vector": VECTOR})
        assert result["backend"] == "arrowspace"

    def test_search_hybrid(self, built_adapter):
        result = built_adapter.search_hybrid("test/ds", {"vector": VECTOR, "alpha": 0.5})
        assert result["backend"] == "arrowspace"

    def test_search_linear_sorted(self, built_adapter):
        result = built_adapter.search_linear_sorted("test/ds", {"vector": VECTOR})
        assert result["backend"] == "arrowspace"

    def test_raises_if_no_index(self, adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            adapter.search("nonexistent/ds", {"vector": VECTOR})


class TestAdapterSpotMethods:
    """spot_motives_eigen / spot_motives_energy / spot_subg_centroids / spot_subg_motives."""

    def test_spot_motives_eigen(self, built_adapter):
        r = built_adapter.spot_motives_eigen("test/ds")
        assert r["method"] == "spot_motives_eigen"
        assert len(r["results"]) == len(FAKE_HITS)

    def test_spot_motives_energy(self, built_adapter):
        r = built_adapter.spot_motives_energy("test/ds")
        assert r["method"] == "spot_motives_energy"

    def test_spot_subg_centroids(self, built_adapter):
        r = built_adapter.spot_subg_centroids("test/ds")
        assert r["method"] == "spot_subg_centroids"

    def test_spot_subg_motives(self, built_adapter):
        r = built_adapter.spot_subg_motives("test/ds")
        assert r["method"] == "spot_subg_motives"


class TestAdapterManifoldStats:
    """manifold_data() and stats_data() convenience helpers."""

    def test_manifold_data_keys(self, built_adapter):
        result = built_adapter.manifold_data("test/ds")
        assert set(result) >= {"nitems", "nfeatures", "nclusters", "lambdas_sorted"}

    def test_manifold_lambdas_sorted_capped_50(self, built_adapter):
        result = built_adapter.manifold_data("test/ds")
        assert len(result["lambdas_sorted"]) <= 50

    def test_stats_data_keys(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert set(result) >= {"nitems", "nfeatures", "nclusters", "gl_nodes", "gl_shape"}

    def test_stats_gl_nodes_positive(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert result["gl_nodes"] > 0

    def test_stats_gl_shape(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert len(result["gl_shape"]) == 2


# ===========================================================================
# 2. LRU cache
# ===========================================================================


class TestLRUIndexCache:
    """Unit tests for the _LRUIndexCache helper."""

    def _make_entry(self):
        from arro_server.arrowspace_adapter import _IndexEntry

        return _IndexEntry(aspace=None, gl=None, nitems=1, nfeatures=1, nclusters=1)

    def test_get_returns_none_on_miss(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        assert cache.get("x") is None

    def test_put_and_get(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        e = self._make_entry()
        cache.put("a", e)
        assert cache.get("a") is e

    def test_evicts_lru_on_overflow(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())
        cache.get("a")  # touch a -> b is now LRU
        cache.put("c", self._make_entry())  # evicts b
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_contains(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        cache.put("x", self._make_entry())
        assert "x" in cache
        assert "y" not in cache

    def test_delete_returns_true(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        cache.put("x", self._make_entry())
        assert cache.delete("x") is True
        assert cache.get("x") is None

    def test_delete_missing_returns_false(self):
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=2)
        assert cache.delete("missing") is False

    def test_eviction_under_maxsize_1(self, fake_mod):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from arro_server.errors import MetadataUnavailable

        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=1)
        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), Path("/tmp/idx"))
        adapter.build_index("ds2", FIXTURE_ARRAY.copy(), Path("/tmp/idx"))
        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("ds1")
        adapter.lambdas("ds2")

    def test_access_refreshes_lru_order(self, fake_mod):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from arro_server.errors import MetadataUnavailable

        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=2)
        adapter.build_index("ds1", FIXTURE_ARRAY.copy(), Path("/tmp/idx"))
        adapter.build_index("ds2", FIXTURE_ARRAY.copy(), Path("/tmp/idx"))
        adapter.lambdas("ds1")  # touch -> MRU
        adapter.build_index("ds3", FIXTURE_ARRAY.copy(), Path("/tmp/idx"))  # evicts ds2
        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("ds2")
        adapter.lambdas("ds1")
        adapter.lambdas("ds3")


# ===========================================================================
# 3. Index lifecycle (HTTP)
# ===========================================================================


class TestIndexLifecycle:
    def test_build_index_200(self, live_client: TestClient):
        r = live_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        body = r.json()
        assert body["built"] is True
        assert body["nitems"] == NITEMS
        assert body["nfeatures"] == NFEATURES
        assert body["nclusters"] == NCLUSTERS

    def test_build_index_flat_graph_params(self, live_client: TestClient):
        """graph_params is a top-level key in the response (not double-nested)."""
        custom = {"eps": 2.0, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
        r = live_client.post(
            f"/api/datasets/{DATASET_ID}/index",
            json={"graph_params": custom},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["graph_params"] == custom
        assert body["nitems"] == NITEMS

    def test_build_index_unknown_dataset_404(self, live_client: TestClient):
        r = live_client.post("/api/datasets/main--missing/index")
        assert r.status_code == 404

    def test_rebuild_index_replaces_cache(self, built_client: TestClient):
        r = built_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        assert r.json()["built"] is True

    # -----------------------------------------------------------------------
    # zzarr persistence assertions (HTTP level)
    #
    # The route passes settings.index_store (ARRO_SERVER_INDEX_STORE) as the
    # index_store argument.  live_client sets that env var to tmp_index_store.
    # _ArrowSpaceAdapter._slug(DATASET_ID) = "main--matrix" (only '/' → '__').
    # Artifacts land at: tmp_index_store / DATASET_SLUG / {data,indices,indptr}.zarr
    # -----------------------------------------------------------------------

    def test_build_index_creates_zzarr_files(self, tmp_index_store: Path, live_client: TestClient):
        """POST /index must create the ArrowSpace index artifacts on disk."""
        live_client.post(f"/api/datasets/{DATASET_ID}/index")
        slug_dir = tmp_index_store / DATASET_SLUG
        assert (slug_dir / "data.zarr").exists(), f"data.zarr missing in {slug_dir}"
        assert (slug_dir / "indices.zarr").exists(), f"indices.zarr missing in {slug_dir}"
        assert (slug_dir / "indptr.zarr").exists(), f"indptr.zarr missing in {slug_dir}"
        assert (slug_dir / "meta.json").exists(), f"meta.json missing in {slug_dir}"

    def test_build_index_meta_json_content(self, tmp_index_store: Path, live_client: TestClient):
        """meta.json written by POST /index must contain nitems and csr_shape.

        Note: graph_params is returned in the HTTP response body by the route
        handler but is NOT written to meta.json by _persist_csr.
        """
        live_client.post(f"/api/datasets/{DATASET_ID}/index")
        slug_dir = tmp_index_store / DATASET_SLUG
        meta = json.loads((slug_dir / "meta.json").read_text())
        assert meta["nitems"] == NITEMS
        assert "csr_shape" in meta, f"expected csr_shape in meta.json, got: {list(meta)}"


# ===========================================================================
# 4. Eigenvalues (HTTP)
# ===========================================================================


class TestLambdas:
    def test_lambdas_200(self, built_client: TestClient):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/lambdas")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["nitems"] == NITEMS
        assert len(body["lambdas"]) == NITEMS
        assert all(isinstance(v, float) for v in body["lambdas"])

    def test_lambdas_sorted_pairs(self, built_client: TestClient):
        body = built_client.get(f"/api/datasets/{DATASET_ID}/lambdas").json()
        for pair in body["lambdas_sorted"]:
            assert len(pair) == 2

    def test_lambdas_no_index_returns_error(self, live_client: TestClient):
        r = live_client.get(f"/api/datasets/{DATASET_ID}/lambdas")
        assert r.status_code in {404, 503}


# ===========================================================================
# 5. Graph Laplacian info (HTTP)
# ===========================================================================


class TestGraphLaplacian:
    def test_graph_laplacian_200(self, built_client: TestClient):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/graph_laplacian")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["nnodes"] == NITEMS
        assert body["shape"] == [NITEMS, NITEMS]
        assert body["graph_params"] == GRAPH_PARAMS

    def test_graph_laplacian_no_index_returns_error(self, live_client: TestClient):
        r = live_client.get(f"/api/datasets/{DATASET_ID}/graph_laplacian")
        assert r.status_code in {404, 503}


# ===========================================================================
# 6. Item retrieval (HTTP)
# ===========================================================================


class TestItemRetrieval:
    def test_get_item_200(self, built_client: TestClient):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/items/0")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["item_index"] == 0
        assert len(body["vector"]) == NFEATURES

    def test_get_item_values_correct(self, built_client: TestClient):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/items/0")
        body = r.json()
        assert body["vector"] == [float(v) for v in FIXTURE_ARRAY[0]]

    def test_get_all_items_200(self, built_client: TestClient):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/items")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["nitems"] == NITEMS
        assert len(body["items"]) == NITEMS
        assert len(body["items"][0]) == NFEATURES

    def test_get_item_no_index_returns_error(self, live_client: TestClient):
        r = live_client.get(f"/api/datasets/{DATASET_ID}/items/0")
        assert r.status_code in {404, 503}


# ===========================================================================
# 7. Search variants (HTTP)
# ===========================================================================


class TestSearchVariants:
    def _post(self, client: TestClient, path: str, body: dict) -> dict:
        r = client.post(f"/api/datasets/{DATASET_ID}/{path}", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_search_spectral(self, built_client: TestClient):
        body = self._post(built_client, "search", {"vector": VECTOR, "tau": 1.0})
        assert body["id"] == DATASET_ID
        assert body["backend"] == "arrowspace"
        assert len(body["results"]) == len(FAKE_HITS)
        assert "index" in body["results"][0]
        assert "score" in body["results"][0]

    def test_search_energy(self, built_client: TestClient):
        body = self._post(built_client, "search/energy", {"vector": VECTOR})
        assert body["backend"] == "arrowspace"

    def test_search_hybrid(self, built_client: TestClient):
        body = self._post(
            built_client, "search/hybrid", {"vector": VECTOR, "tau": 1.0, "alpha": 0.5}
        )
        assert body["backend"] == "arrowspace"

    def test_search_hybrid_alpha_zero(self, built_client: TestClient):
        body = self._post(built_client, "search/hybrid", {"vector": VECTOR, "alpha": 0.0})
        assert body["backend"] == "arrowspace"

    def test_search_hybrid_alpha_one(self, built_client: TestClient):
        body = self._post(built_client, "search/hybrid", {"vector": VECTOR, "alpha": 1.0})
        assert body["backend"] == "arrowspace"

    def test_search_linear(self, built_client: TestClient):
        body = self._post(built_client, "search/linear", {"vector": VECTOR})
        assert body["backend"] == "arrowspace"

    def test_search_batch(self, built_client: TestClient):
        body = self._post(built_client, "search/batch", {"vectors": [VECTOR, VECTOR], "tau": 1.0})
        assert body["backend"] == "arrowspace"
        assert len(body["results"]) == 2
        for result_list in body["results"]:
            assert len(result_list) == len(FAKE_HITS)

    def test_search_missing_vector_422(self, built_client: TestClient):
        """Pydantic model on POST /search returns 422 for missing field."""
        r = built_client.post(f"/api/datasets/{DATASET_ID}/search", json={"tau": 1.0})
        assert r.status_code == 422

    def test_search_batch_missing_vectors_422(self, built_client: TestClient):
        r = built_client.post(f"/api/datasets/{DATASET_ID}/search/batch", json={"tau": 1.0})
        assert r.status_code == 422

    def test_search_wrong_vector_type_422(self, built_client: TestClient):
        """Pydantic rejects string where list[float] expected."""
        r = built_client.post(
            f"/api/datasets/{DATASET_ID}/search",
            json={"vector": "not-a-list"},
        )
        assert r.status_code == 422

    def test_search_no_index_returns_error(self, live_client: TestClient):
        r = live_client.post(f"/api/datasets/{DATASET_ID}/search", json={"vector": VECTOR})
        assert r.status_code in {404, 503}


# ===========================================================================
# 8. Spot methods (HTTP)
# ===========================================================================


SPOT_ENDPOINTS = [
    ("spot/motives/eigen", "spot_motives_eigen"),
    ("spot/motives/energy", "spot_motives_energy"),
    ("spot/subgraphs/centroids", "spot_subg_centroids"),
    ("spot/subgraphs/motives", "spot_subg_motives"),
]


class TestSpotMethods:
    @pytest.mark.parametrize("endpoint,method_name", SPOT_ENDPOINTS)
    def test_spot_200(self, built_client: TestClient, endpoint: str, method_name: str):
        r = built_client.get(f"/api/datasets/{DATASET_ID}/{endpoint}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["method"] == method_name
        assert len(body["results"]) == len(FAKE_HITS)

    @pytest.mark.parametrize("endpoint,_", SPOT_ENDPOINTS)
    def test_spot_result_schema(self, built_client: TestClient, endpoint: str, _: str):
        body = built_client.get(f"/api/datasets/{DATASET_ID}/{endpoint}").json()
        for result in body["results"]:
            assert isinstance(result["index"], int)
            assert isinstance(result["score"], float)

    @pytest.mark.parametrize("endpoint,_", SPOT_ENDPOINTS)
    def test_spot_no_index(self, live_client: TestClient, endpoint: str, _: str):
        r = live_client.get(f"/api/datasets/{DATASET_ID}/{endpoint}")
        assert r.status_code in {404, 503}


# ===========================================================================
# 9. Error cases (HTTP)
# ===========================================================================


class TestErrorCases:
    def test_index_1d_array_raises(self, live_client: TestClient):
        """1-D dataset triggers ValueError in build_index -> 400/422/500."""
        r = live_client.post("/api/datasets/main--vector/index")
        assert r.status_code in {400, 422, 500}

    def test_unknown_dataset_404_on_all_endpoints(self, built_client: TestClient):
        endpoints = [
            ("GET", "/api/datasets/main--missing/graph_laplacian"),
            ("GET", "/api/datasets/main--missing/items"),
            ("GET", "/api/datasets/main--missing/items/0"),
            ("GET", "/api/datasets/main--missing/spot/motives/eigen"),
            ("GET", "/api/datasets/main--missing/spot/motives/energy"),
            ("GET", "/api/datasets/main--missing/spot/subgraphs/centroids"),
            ("GET", "/api/datasets/main--missing/spot/subgraphs/motives"),
        ]
        for method, path in endpoints:
            r = built_client.request(method, path)
            assert r.status_code == 404, f"{method} {path} -> {r.status_code}"


# ===========================================================================
# 10. Sidecar fallback
# ===========================================================================


class TestSidecarFallback:
    @pytest.fixture
    def sidecar_client(self, tmp_zarr_root: Path):
        from arro_server import arrowspace_adapter
        from arro_server import settings as settings_mod
        from arro_server.app import create_app
        from arro_server.storage import registry as registry_mod

        original_load = arrowspace_adapter.load

        def patched_load():
            return arrowspace_adapter._SidecarAdapter()

        arrowspace_adapter.load = patched_load  # type: ignore[assignment]

        os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
        os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
        settings_mod.reset_settings_cache()
        registry_mod.reset_registry_cache()
        arrowspace_adapter.reset_adapter_cache()

        app = create_app()
        with TestClient(app) as client:
            yield client

        arrowspace_adapter.load = original_load  # type: ignore[assignment]
        os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
        os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
        settings_mod.reset_settings_cache()
        registry_mod.reset_registry_cache()
        arrowspace_adapter.reset_adapter_cache()

    def test_build_index_503_without_package(self, sidecar_client: TestClient):
        r = sidecar_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 503

    def test_lambdas_503_without_package(self, sidecar_client: TestClient):
        r = sidecar_client.get(f"/api/datasets/{DATASET_ID}/lambdas")
        assert r.status_code == 503

    def test_search_503_without_package(self, sidecar_client: TestClient):
        r = sidecar_client.post(
            f"/api/datasets/{DATASET_ID}/search",
            json={"vector": VECTOR},
        )
        assert r.status_code == 503

    def test_search_energy_503_without_package(self, sidecar_client: TestClient):
        r = sidecar_client.post(
            f"/api/datasets/{DATASET_ID}/search/energy",
            json={"vector": VECTOR},
        )
        assert r.status_code == 503

    def test_search_batch_503_without_package(self, sidecar_client: TestClient):
        r = sidecar_client.post(
            f"/api/datasets/{DATASET_ID}/search/batch",
            json={"vectors": [VECTOR]},
        )
        assert r.status_code == 503

    def test_sidecar_manifold_still_works(self, sidecar_client: TestClient):
        r = sidecar_client.get(f"/api/datasets/{DATASET_ID}/manifold")
        assert r.status_code == 200

    def test_sidecar_keyword_search_still_works(self, sidecar_client: TestClient):
        r = sidecar_client.get(f"/api/datasets/{DATASET_ID}/search?q=alpha")
        assert r.status_code == 200
        assert r.json()["results"][0]["id"] == "row-0"
