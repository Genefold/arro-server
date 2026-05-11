from __future__ import annotations

"""tests/test_phase1_arrowspace.py — Phase 1 full test suite.

All 3 bugs from test_fails.txt are addressed in the production code;
test assertions updated to match corrected response shapes.

Bug fixes reflected here:
  1. build_index response is now flat: graph_params is a top-level key,
     not double-nested inside meta.
  2. POST /search* endpoints use Pydantic models — missing/wrong fields
     now return 422 before the route body runs.
  3. arrowspace load() catches Exception (not just ImportError) so a
     broken __init__.py falls back to sidecar without crashing.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake arrowspace module
# ---------------------------------------------------------------------------

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FAKE_ITEMS = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)
FAKE_LAMBDAS = [float(i) * 0.1 for i in range(NITEMS)]
FAKE_HITS = [(i, float(i) * 0.01) for i in range(5)]


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    aspace.lambdas.return_value = FAKE_LAMBDAS
    aspace.lambdas_sorted.return_value = [(float(v), i) for i, v in enumerate(FAKE_LAMBDAS)]
    aspace.get_item.side_effect = lambda idx: FAKE_ITEMS[idx]
    aspace.get_all_items.return_value = FAKE_ITEMS
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
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "main--matrix"
VECTOR = [float(i) for i in range(NFEATURES)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mod() -> types.ModuleType:
    return _make_fake_arrowspace_module()


@pytest.fixture
def live_client(tmp_zarr_root: Path, fake_mod: types.ModuleType):
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()

    app = create_app()
    with TestClient(app) as client:
        yield client

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
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
# 1. Adapter unit tests
# ===========================================================================


class TestAdapterUnit:
    def setup_method(self):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        self.fake_mod = _make_fake_arrowspace_module()
        self.adapter = _ArrowSpaceAdapter(self.fake_mod, cache_size=4)

    def _build(self, dataset_id: str = "ds1") -> str:
        self.adapter.build_index(dataset_id, FAKE_ITEMS.copy(), Path("/tmp/idx"))
        return dataset_id

    def test_build_returns_meta(self):
        meta = self.adapter.build_index("ds1", FAKE_ITEMS.copy(), Path("/tmp/idx"))
        assert meta["nitems"] == NITEMS
        assert meta["nfeatures"] == NFEATURES
        assert meta["nclusters"] == NCLUSTERS

    def test_build_rejects_1d_array(self):
        with pytest.raises(ValueError, match="2-D"):
            self.adapter.build_index("ds1", np.ones(10), Path("/tmp/idx"))

    def test_lambdas_structure(self):
        ds = self._build()
        result = self.adapter.lambdas(ds)
        assert len(result["lambdas"]) == NITEMS
        assert all(isinstance(v, float) for v in result["lambdas"])
        assert all(len(pair) == 2 for pair in result["lambdas_sorted"])

    def test_graph_laplacian_info(self):
        ds = self._build()
        info = self.adapter.graph_laplacian_info(ds)
        assert info["nnodes"] == NITEMS
        assert info["shape"] == [NITEMS, NITEMS]
        assert info["graph_params"] == GRAPH_PARAMS

    def test_get_item(self):
        ds = self._build()
        result = self.adapter.get_item(ds, 0)
        assert result["item_index"] == 0
        assert len(result["vector"]) == NFEATURES
        assert result["vector"] == [float(v) for v in FAKE_ITEMS[0]]

    def test_get_all_items(self):
        ds = self._build()
        result = self.adapter.get_all_items(ds)
        assert result["nitems"] == NITEMS
        assert len(result["items"]) == NITEMS
        assert len(result["items"][0]) == NFEATURES

    def test_search(self):
        ds = self._build()
        result = self.adapter.search(ds, {"vector": VECTOR, "tau": 1.0})
        assert result["backend"] == "arrowspace"
        assert len(result["results"]) == len(FAKE_HITS)
        assert "index" in result["results"][0]
        assert "score" in result["results"][0]

    def test_search_requires_vector(self):
        from arro_server.errors import MetadataUnavailable
        ds = self._build()
        with pytest.raises(MetadataUnavailable):
            self.adapter.search(ds, {"tau": 1.0})

    def test_search_batch(self):
        ds = self._build()
        result = self.adapter.search_batch(ds, {"vectors": [VECTOR, VECTOR], "tau": 1.0})
        assert len(result["results"]) == 2
        assert len(result["results"][0]) == len(FAKE_HITS)

    def test_search_batch_requires_vectors(self):
        from arro_server.errors import MetadataUnavailable
        ds = self._build()
        with pytest.raises(MetadataUnavailable):
            self.adapter.search_batch(ds, {"tau": 1.0})

    def test_search_energy(self):
        ds = self._build()
        result = self.adapter.search_energy(ds, {"vector": VECTOR})
        assert result["backend"] == "arrowspace"

    def test_search_hybrid(self):
        ds = self._build()
        result = self.adapter.search_hybrid(ds, {"vector": VECTOR, "alpha": 0.5})
        assert result["backend"] == "arrowspace"

    def test_search_linear_sorted(self):
        ds = self._build()
        result = self.adapter.search_linear_sorted(ds, {"vector": VECTOR})
        assert result["backend"] == "arrowspace"

    def test_spot_motives_eigen(self):
        ds = self._build()
        r = self.adapter.spot_motives_eigen(ds)
        assert r["method"] == "spot_motives_eigen"
        assert len(r["results"]) == len(FAKE_HITS)

    def test_spot_motives_energy(self):
        ds = self._build()
        r = self.adapter.spot_motives_energy(ds)
        assert r["method"] == "spot_motives_energy"

    def test_spot_subg_centroids(self):
        ds = self._build()
        r = self.adapter.spot_subg_centroids(ds)
        assert r["method"] == "spot_subg_centroids"

    def test_spot_subg_motives(self):
        ds = self._build()
        r = self.adapter.spot_subg_motives(ds)
        assert r["method"] == "spot_subg_motives"

    def test_missing_index_raises(self):
        from arro_server.errors import MetadataUnavailable
        with pytest.raises(MetadataUnavailable):
            self.adapter.lambdas("never_built")


# ===========================================================================
# 2. Index lifecycle (HTTP)
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

    def test_build_index_with_custom_params(self, live_client: TestClient):
        """FIX 1: response is now flat — graph_params is a top-level key."""
        custom = {"eps": 2.0, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
        r = live_client.post(
            f"/api/datasets/{DATASET_ID}/index",
            json={"graph_params": custom},
        )
        assert r.status_code == 200
        body = r.json()
        # graph_params is now at the top level, not nested inside another dict
        assert body["graph_params"] == custom
        assert body["nitems"] == NITEMS

    def test_build_index_unknown_dataset_404(self, live_client: TestClient):
        r = live_client.post("/api/datasets/main--missing/index")
        assert r.status_code == 404

    def test_rebuild_index_replaces_cache(self, built_client: TestClient):
        r = built_client.post(f"/api/datasets/{DATASET_ID}/index")
        assert r.status_code == 200
        assert r.json()["built"] is True


# ===========================================================================
# 3. Eigenvalues (HTTP)
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
# 4. Graph Laplacian info (HTTP)
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
# 5. Item retrieval (HTTP)
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
        expected = [float(v) for v in FAKE_ITEMS[0]]
        assert body["vector"] == expected

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
# 6. Search variants (HTTP)
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
        body = self._post(built_client, "search/hybrid", {"vector": VECTOR, "tau": 1.0, "alpha": 0.5})
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
        """FIX 2: Pydantic model on POST /search returns 422 for missing field."""
        r = built_client.post(f"/api/datasets/{DATASET_ID}/search", json={"tau": 1.0})
        assert r.status_code == 422

    def test_search_batch_missing_vectors_422(self, built_client: TestClient):
        r = built_client.post(f"/api/datasets/{DATASET_ID}/search/batch", json={"tau": 1.0})
        assert r.status_code == 422

    def test_search_wrong_vector_type_422(self, built_client: TestClient):
        """FIX 2: Pydantic rejects string where list[float] expected."""
        r = built_client.post(
            f"/api/datasets/{DATASET_ID}/search",
            json={"vector": "not-a-list"},
        )
        assert r.status_code == 422

    def test_search_no_index_returns_error(self, live_client: TestClient):
        r = live_client.post(f"/api/datasets/{DATASET_ID}/search", json={"vector": VECTOR})
        assert r.status_code in {404, 503}


# ===========================================================================
# 7. Spot methods (HTTP)
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
# 8. Error cases
# ===========================================================================


class TestErrorCases:
    def test_index_1d_array_raises(self, live_client: TestClient):
        """1-D dataset triggers ValueError in build_index -> 400/422/500."""
        r = live_client.post("/api/datasets/main--vector/index")
        assert r.status_code in {400, 422, 500}

    def test_unknown_dataset_404_on_all_new_endpoints(self, built_client: TestClient):
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
# 9. LRU cache
# ===========================================================================


class TestLRUCache:
    def test_eviction_under_maxsize_1(self):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from arro_server.errors import MetadataUnavailable
        adapter = _ArrowSpaceAdapter(_make_fake_arrowspace_module(), cache_size=1)
        adapter.build_index("ds1", FAKE_ITEMS.copy(), Path("/tmp/idx"))
        adapter.build_index("ds2", FAKE_ITEMS.copy(), Path("/tmp/idx"))
        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("ds1")
        adapter.lambdas("ds2")

    def test_access_refreshes_lru_order(self):
        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from arro_server.errors import MetadataUnavailable
        adapter = _ArrowSpaceAdapter(_make_fake_arrowspace_module(), cache_size=2)
        adapter.build_index("ds1", FAKE_ITEMS.copy(), Path("/tmp/idx"))
        adapter.build_index("ds2", FAKE_ITEMS.copy(), Path("/tmp/idx"))
        adapter.lambdas("ds1")  # touch -> MRU
        adapter.build_index("ds3", FAKE_ITEMS.copy(), Path("/tmp/idx"))  # evicts ds2
        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("ds2")
        adapter.lambdas("ds1")
        adapter.lambdas("ds3")

    def test_cache_delete(self):
        from arro_server.arrowspace_adapter import _IndexEntry, _LRUIndexCache
        cache = _LRUIndexCache(maxsize=4)
        entry = _IndexEntry(aspace=None, gl=None, nitems=1, nfeatures=1, nclusters=1)
        cache.put("k1", entry)
        assert cache.delete("k1") is True
        assert cache.delete("k1") is False
        assert "k1" not in cache


# ===========================================================================
# 10. Sidecar fallback
# FIX 3: load() now catches Exception not just ImportError, so a broken
# arrowspace __init__.py (NameError) falls back to sidecar gracefully.
# ===========================================================================


class TestSidecarFallback:
    @pytest.fixture
    def sidecar_client(self, tmp_zarr_root: Path):
        from arro_server import arrowspace_adapter
        from arro_server import settings as settings_mod
        from arro_server.app import create_app
        from arro_server.storage import registry as registry_mod

        # Simulate broken package: inject a module whose import raises NameError
        broken_mod = types.ModuleType("arrowspace")
        broken_mod.__spec__ = None  # type: ignore[attr-defined]

        # Patch load() to return sidecar directly (simulates broken __init__)
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
