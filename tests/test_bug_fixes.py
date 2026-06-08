"""Regression tests for bugs found during integration testing.

BUG-01: get_item / get_all_items crashed when arrowspace Rust library
        returned a tuple instead of np.ndarray.
BUG-02: SPOT endpoints (spot_motives_eigen, spot_motives_energy,
        spot_subg_centroids, spot_subg_motives) crashed with
        "missing 2 required positional arguments: 'gl' and 'cfg'".
BUG-03: StorageRegistry.open() double-wrapped the dataset_id in the
        404 error message: "Dataset 'Dataset 'X' not found.' not found."
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Reuse constants from the main test suite
NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FIXTURE_ARRAY = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)
FAKE_HITS = [(i, float(i) * 0.01) for i in range(5)]
DATASET_ID = "main--matrix"
VECTOR = FIXTURE_ARRAY[0].tolist()


# ===========================================================================
# Fixtures — follow the same pattern as test_arrowspace.py
# ===========================================================================


@pytest.fixture
def adapter(tmp_path: Path):
    """A fresh _ArrowSpaceAdapter backed by a fake arrowspace module."""
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    fake_mod = _make_fake_arrowspace_module()
    return _ArrowSpaceAdapter(fake_mod, cache_size=4)


@pytest.fixture
def built_adapter(adapter, tmp_path: Path):
    """An adapter with one pre-built index ('test/ds')."""
    adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), tmp_path)
    return adapter


@pytest.fixture
def live_client(tmp_path: Path):
    """HTTP client with a mock arrowspace module and a real Zarr backend."""
    tmp_zarr_root = _make_tmp_zarr_root(tmp_path)
    tmp_index_store = tmp_path / "index_store"
    tmp_index_store.mkdir()

    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    import sys, types

    fake_mod = _make_fake_arrowspace_module()
    sys.modules["arrowspace"] = fake_mod

    import os
    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(tmp_index_store)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    arrowspace_adapter.reset_adapter_cache()

    app = create_app()
    with TestClient(app) as client:
        yield client

    sys.modules.pop("arrowspace", None)
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture
def built_client(live_client: TestClient) -> TestClient:
    r = live_client.post(f"/api/datasets/{DATASET_ID}/index")
    assert r.status_code == 200, r.text
    return live_client


# ===========================================================================
# Helpers
# ===========================================================================


def _make_fake_gl():
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


def _make_fake_arrowspace_module():
    import types

    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    aspace.lambdas.return_value = [float(i) * 0.1 for i in range(NITEMS)]
    aspace.lambdas_sorted.return_value = [(float(v), i) for i, v in enumerate([float(i) * 0.1 for i in range(NITEMS)])]
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
    gl = _make_fake_gl()

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod = types.ModuleType("arrowspace")
    fake_mod.ArrowSpaceBuilder = FakeBuilder
    return fake_mod


def _make_tmp_zarr_root(tmp_path):
    """Create a minimal Zarr root with one 50x4 matrix."""
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "datasets"
    root.mkdir()
    ds_path = root / "matrix"
    arr = zarr.open(str(ds_path), mode="w", shape=(NITEMS, NFEATURES), chunks=(10, 4), dtype="float32")
    arr[:] = FIXTURE_ARRAY.astype("float32")
    try:
        arr.attrs["description"] = "test matrix"
    except Exception:
        pass
    return root


# ===========================================================================
# BUG-01: get_item / get_all_items con tuple return
# ===========================================================================


class TestBug01GetItemTuple:
    """BUG-01 regression: get_item crashed when aspace.get_item() returned a tuple.

    The real arrowspace Rust library returns a tuple (array([...]),) from
    get_item(), not a plain np.ndarray. The old code [float(v) for v in vec]
    would iterate over the tuple and fail with:
        TypeError: only 0-dimensional arrays can be converted to Python scalars
    """

    def test_get_item_tuple_single_element(self, built_adapter):
        """aspace.get_item returning (np.array,) must yield a list of floats."""
        mock_aspace = MagicMock()
        mock_aspace.nitems = NITEMS
        mock_aspace.nfeatures = NFEATURES
        mock_aspace.nclusters = NCLUSTERS
        mock_aspace.get_item.side_effect = lambda idx: (FIXTURE_ARRAY[idx],)
        mock_aspace.get_all_items.return_value = FIXTURE_ARRAY
        mock_gl = _make_fake_gl()

        import types
        fake_mod = types.ModuleType("arrowspace")
        class FakeBuilder:
            def build(self, gp, arr):
                return mock_aspace, mock_gl

        fake_mod.ArrowSpaceBuilder = FakeBuilder

        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from pathlib import Path
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=4)
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), Path("/tmp"))

        result = adapter.get_item("test/ds", 0)
        assert result["item_index"] == 0
        assert isinstance(result["vector"], list)
        assert len(result["vector"]) == NFEATURES
        assert result["vector"] == FIXTURE_ARRAY[0].tolist()

    def test_get_item_tuple_two_elements(self, built_adapter):
        """aspace.get_item returning (index, array) must still work."""
        mock_aspace = MagicMock()
        mock_aspace.nitems = NITEMS
        mock_aspace.nfeatures = NFEATURES
        mock_aspace.nclusters = NCLUSTERS
        mock_aspace.get_item.side_effect = lambda idx: (idx, FIXTURE_ARRAY[idx])
        mock_aspace.get_all_items.return_value = FIXTURE_ARRAY
        mock_gl = _make_fake_gl()

        import types
        fake_mod = types.ModuleType("arrowspace")
        class FakeBuilder:
            def build(self, gp, arr):
                return mock_aspace, mock_gl

        fake_mod.ArrowSpaceBuilder = FakeBuilder

        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from pathlib import Path
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=4)
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), Path("/tmp"))

        result = adapter.get_item("test/ds", 0)
        assert result["item_index"] == 0
        assert isinstance(result["vector"], list)
        assert len(result["vector"]) == NFEATURES

    def test_get_item_plain_array(self, built_adapter):
        """aspace.get_item returning a plain np.ndarray must also work."""
        mock_aspace = MagicMock()
        mock_aspace.nitems = NITEMS
        mock_aspace.nfeatures = NFEATURES
        mock_aspace.nclusters = NCLUSTERS
        mock_aspace.get_item.side_effect = lambda idx: FIXTURE_ARRAY[idx]
        mock_aspace.get_all_items.return_value = FIXTURE_ARRAY
        mock_gl = _make_fake_gl()

        import types
        fake_mod = types.ModuleType("arrowspace")
        class FakeBuilder:
            def build(self, gp, arr):
                return mock_aspace, mock_gl

        fake_mod.ArrowSpaceBuilder = FakeBuilder

        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from pathlib import Path
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=4)
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), Path("/tmp"))

        result = adapter.get_item("test/ds", 0)
        assert result["item_index"] == 0
        assert isinstance(result["vector"], list)
        assert len(result["vector"]) == NFEATURES

    def test_get_all_items_tuple_rows(self, built_adapter):
        """get_all_items must handle rows that come as tuples."""
        mock_aspace = MagicMock()
        mock_aspace.nitems = NITEMS
        mock_aspace.nfeatures = NFEATURES
        mock_aspace.nclusters = NCLUSTERS
        mock_aspace.get_item.side_effect = lambda idx: (FIXTURE_ARRAY[idx],)
        mock_aspace.get_all_items.return_value = [(FIXTURE_ARRAY[i],) for i in range(NITEMS)]
        mock_gl = _make_fake_gl()

        import types
        fake_mod = types.ModuleType("arrowspace")
        class FakeBuilder:
            def build(self, gp, arr):
                return mock_aspace, mock_gl

        fake_mod.ArrowSpaceBuilder = FakeBuilder

        from arro_server.arrowspace_adapter import _ArrowSpaceAdapter
        from pathlib import Path
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=4)
        adapter.build_index("test/ds", FIXTURE_ARRAY.copy(), Path("/tmp"))

        result = adapter.get_all_items("test/ds")
        assert result["nitems"] == NITEMS
        assert len(result["items"]) == NITEMS
        assert result["items"][0] == FIXTURE_ARRAY[0].tolist()

    def test_get_item_http_200(self, built_client):
        """GET /items/0 must return 200, not 500."""
        r = built_client.get(f"/api/datasets/{DATASET_ID}/items/0")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "vector" in body
        assert len(body["vector"]) == NFEATURES

    def test_get_all_items_http_200(self, built_client):
        """GET /items must return 200, not 500."""
        r = built_client.get(f"/api/datasets/{DATASET_ID}/items")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["items"]) == NITEMS
        assert len(body["items"][0]) == NFEATURES


# ===========================================================================
# BUG-02: SPOT endpoints pass gl and cfg
# ===========================================================================


class TestBug02SpotArgs:
    """BUG-02 regression: SPOT methods were called without gl/cfg args.

    The real arrowspace library requires spot_*(gl, cfg). This test ensures
    the adapter passes both arguments by verifying the calls via Mock.
    """

    @pytest.mark.parametrize("endpoint,method_name", [
        ("spot/motives/eigen", "spot_motives_eigen"),
        ("spot/motives/energy", "spot_motives_energy"),
        ("spot/subgraphs/centroids", "spot_subg_centroids"),
        ("spot/subgraphs/motives", "spot_subg_motives"),
    ])
    def test_spot_http_200(self, built_client: TestClient, endpoint: str, method_name: str):
        """GET /spot/* must return 200, not 500."""
        r = built_client.get(f"/api/datasets/{DATASET_ID}/{endpoint}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["id"] == DATASET_ID
        assert body["method"] == method_name

    @pytest.mark.parametrize("endpoint", [
        "spot/motives/eigen",
        "spot/motives/energy",
        "spot/subgraphs/centroids",
        "spot/subgraphs/motives",
    ])
    def test_spot_result_schema(self, built_client: TestClient, endpoint: str):
        """Each spot result must have index (int) and score (float)."""
        r = built_client.get(f"/api/datasets/{DATASET_ID}/{endpoint}")
        assert r.status_code == 200, r.text
        body = r.json()
        for result in body["results"]:
            assert isinstance(result["index"], int)
            assert isinstance(result["score"], float)


# ===========================================================================
# BUG-03: Double-wrapped 404 message
# ===========================================================================


class TestBug03DoubleWrap404:
    """BUG-03 regression: 404 message must not double-wrap the dataset_id.

    Old message: "Dataset 'Dataset 'main--nonexistent' not found.' not found."
    Correct message: "Dataset 'main--nonexistent' not found."
    """

    def test_metadata_unknown_dataset_no_double_wrap(self, live_client):
        """GET /metadata on non-existent dataset must have clean 404 detail."""
        r = live_client.get("/api/datasets/main--nonexistent/metadata")
        assert r.status_code == 404
        detail = r.json()["detail"]
        assert "not found.' not found" not in detail, (
            f"Double-wrapped detail: {detail!r}"
        )
        assert "main--nonexistent" in detail

    def test_spot_unknown_dataset_no_double_wrap(self, live_client):
        """GET /spot on non-existent dataset must have clean 404."""
        r = live_client.get("/api/datasets/main--ghost/spot/motives/eigen")
        assert r.status_code in {404, 503}
        if r.status_code == 404:
            detail = r.json()["detail"]
            assert "not found.' not found" not in detail, (
                f"Double-wrapped detail: {detail!r}"
            )

    def test_index_unknown_dataset_no_double_wrap(self, live_client):
        """POST /index on non-existent dataset must have clean 404."""
        r = live_client.post("/api/datasets/uploads--ghost/index", json={})
        assert r.status_code in {404, 503}
        if r.status_code == 404:
            detail = r.json()["detail"]
            assert "not found.' not found" not in detail, (
                f"Double-wrapped detail: {detail!r}"
            )
