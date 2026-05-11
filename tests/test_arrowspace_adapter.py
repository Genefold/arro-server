"""Unit tests for the live _ArrowSpaceAdapter.

All tests in this module are skipped automatically when the ``arrowspace``
package is not installed (pytest.importorskip).

Fixture dataset: 30 rows x 8 features, random float64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if arrowspace is not installed
arrowspace = pytest.importorskip("arrowspace")

from arro_server.arrowspace_adapter import (  # noqa: E402
    DEFAULT_GRAPH_PARAMS,  # noqa: F401
    _ArrowSpaceAdapter,
    _IndexEntry,
    _LRUIndexCache,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_ITEMS = 30
N_FEATURES = 8

# Fixed seed for reproducibility
FIXTURE_ARRAY = np.random.default_rng(42).random((N_ITEMS, N_FEATURES)).astype(np.float64)


@pytest.fixture
def adapter():
    """A fresh _ArrowSpaceAdapter backed by the real arrowspace package."""
    return _ArrowSpaceAdapter(arrowspace, cache_size=4)


@pytest.fixture
def built_adapter(adapter, tmp_path: Path):
    """An adapter with one pre-built index ('test/ds')."""
    adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
    return adapter


# ---------------------------------------------------------------------------
# Task 1.2 — _ArrowSpaceAdapter correctness
# ---------------------------------------------------------------------------


class TestBuildIndex:
    """Task 1.2 / 1.3 — build_index builds and caches a valid index."""

    def test_returns_expected_keys(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        assert set(meta) >= {"nitems", "nfeatures", "nclusters"}

    def test_nitems_matches_array(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        assert meta["nitems"] == N_ITEMS

    def test_nfeatures_matches_array(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        assert meta["nfeatures"] == N_FEATURES

    def test_nclusters_is_positive(self, adapter, tmp_path):
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        assert meta["nclusters"] > 0

    def test_entry_cached(self, adapter, tmp_path):
        adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        assert adapter._cache.get("test/ds") is not None

    def test_rejects_1d_array(self, adapter, tmp_path):
        with pytest.raises(ValueError, match="2-D"):
            adapter.build_index("test/ds", np.ones(10), tmp_path)

    def test_custom_graph_params(self, adapter, tmp_path):
        custom = {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
        meta = adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path, graph_params=custom)
        assert meta["nitems"] == N_ITEMS

    def test_csr_files_persisted(self, adapter, tmp_path):
        """CSR zarr arrays should be written to disk after build."""
        adapter.build_index("test/ds", FIXTURE_ARRAY, tmp_path)
        slug_dir = tmp_path / "test__ds"
        assert (slug_dir / "data.zarr").exists()
        assert (slug_dir / "indices.zarr").exists()
        assert (slug_dir / "indptr.zarr").exists()
        assert (slug_dir / "meta.json").exists()


class TestLambdas:
    """Task 1.2 — lambdas() returns valid eigenvalue data."""

    def test_returns_expected_keys(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert set(result) >= {"nitems", "lambdas", "lambdas_sorted"}

    def test_lambdas_is_list_of_floats(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert isinstance(result["lambdas"], list)
        assert all(isinstance(v, float) for v in result["lambdas"])

    def test_lambdas_sorted_structure(self, built_adapter):
        """lambdas_sorted must be a list of [float, int] pairs."""
        result = built_adapter.lambdas("test/ds")
        for pair in result["lambdas_sorted"]:
            assert len(pair) == 2
            assert isinstance(pair[0], float)
            assert isinstance(pair[1], int)

    def test_lambdas_not_empty(self, built_adapter):
        result = built_adapter.lambdas("test/ds")
        assert len(result["lambdas"]) > 0

    def test_raises_404_if_no_index(self, adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            adapter.lambdas("nonexistent/ds")


class TestSearch:
    """Task 1.2 / 1.4 — search() against a built index."""

    def test_returns_results_list(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": FIXTURE_ARRAY[0].tolist()})
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_results_have_index_and_score(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": FIXTURE_ARRAY[0].tolist()})
        assert len(result["results"]) > 0
        hit = result["results"][0]
        assert "index" in hit
        assert "score" in hit

    def test_index_is_int_score_is_float(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": FIXTURE_ARRAY[0].tolist()})
        hit = result["results"][0]
        assert isinstance(hit["index"], int)
        assert isinstance(hit["score"], float)

    def test_backend_field_is_arrowspace(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": FIXTURE_ARRAY[0].tolist()})
        assert result["backend"] == "arrowspace"

    def test_custom_tau(self, built_adapter):
        result = built_adapter.search("test/ds", {"vector": FIXTURE_ARRAY[0].tolist(), "tau": 2.0})
        assert "results" in result

    def test_raises_404_missing_vector(self, built_adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable, match="vector"):
            built_adapter.search("test/ds", {})

    def test_raises_404_if_no_index(self, adapter):
        from arro_server.errors import MetadataUnavailable

        with pytest.raises(MetadataUnavailable):
            adapter.search("nonexistent/ds", {"vector": [0.1, 0.2]})


class TestManifoldData:
    """Task 1.2 / 1.5 — manifold_data() returns live topology summary."""

    def test_returns_expected_keys(self, built_adapter):
        result = built_adapter.manifold_data("test/ds")
        assert set(result) >= {"nitems", "nfeatures", "nclusters", "lambdas_sorted"}

    def test_lambdas_sorted_capped_at_50(self, built_adapter):
        result = built_adapter.manifold_data("test/ds")
        assert len(result["lambdas_sorted"]) <= 50

    def test_nitems_correct(self, built_adapter):
        result = built_adapter.manifold_data("test/ds")
        assert result["nitems"] == N_ITEMS


class TestStatsData:
    """Task 1.2 / 1.5 — stats_data() returns GraphLaplacian stats."""

    def test_returns_expected_keys(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert set(result) >= {"nitems", "nfeatures", "nclusters", "gl_nodes", "gl_shape"}

    def test_gl_nodes_is_positive(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert result["gl_nodes"] > 0

    def test_gl_shape_is_two_element_list(self, built_adapter):
        result = built_adapter.stats_data("test/ds")
        assert len(result["gl_shape"]) == 2


# ---------------------------------------------------------------------------
# LRU cache correctness
# ---------------------------------------------------------------------------


class TestLRUIndexCache:
    """Unit tests for the _LRUIndexCache helper."""

    def _make_entry(self) -> _IndexEntry:
        return _IndexEntry(aspace=None, gl=None, nitems=1, nfeatures=1, nclusters=1)

    def test_get_returns_none_on_miss(self):
        cache = _LRUIndexCache(maxsize=2)
        assert cache.get("x") is None

    def test_put_and_get(self):
        cache = _LRUIndexCache(maxsize=2)
        e = self._make_entry()
        cache.put("a", e)
        assert cache.get("a") is e

    def test_evicts_oldest_on_overflow(self):
        cache = _LRUIndexCache(maxsize=2)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())
        cache.get("a")  # touch a -> b is now LRU
        cache.put("c", self._make_entry())  # should evict b
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_contains(self):
        cache = _LRUIndexCache(maxsize=2)
        cache.put("x", self._make_entry())
        assert "x" in cache
        assert "y" not in cache

    def test_delete(self):
        cache = _LRUIndexCache(maxsize=2)
        cache.put("x", self._make_entry())
        assert cache.delete("x") is True
        assert cache.get("x") is None

    def test_delete_missing_returns_false(self):
        cache = _LRUIndexCache(maxsize=2)
        assert cache.delete("missing") is False
