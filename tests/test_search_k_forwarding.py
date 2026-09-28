"""taumode search must forward the request's k to the library.

Library contract: search(item, gl, tau, k=None) on arrowspace>=0.28.1 (the
declared minimum). The adapter forwards explicit k directly; k absent keeps
the index's topk. A real TypeError from inside search is never relabelled.
"""

from __future__ import annotations

import importlib.metadata
from types import ModuleType
from unittest.mock import MagicMock

import arrowspace
import numpy as np
import pytest
from packaging.version import Version

from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _IndexEntry

DATASET_ID = "test/ds"

MIN_ARROWSPACE = Version("0.28.1")


def make_adapter(tmp_path, search_impl=None) -> tuple[_ArrowSpaceAdapter, MagicMock]:
    aspace = MagicMock()
    aspace.nitems = 2
    aspace.nfeatures = 2
    aspace.nclusters = 1
    aspace.lambdas.return_value = []
    aspace.lambdas_sorted.return_value = []
    aspace.search.return_value = [(0, 1.0), (1, 0.9)]
    if search_impl is not None:
        aspace.search = search_impl
    gl = MagicMock()
    gl.nnodes = 2
    gl.to_csr.return_value = (
        np.ones(2, dtype=np.float32),
        np.arange(2, dtype=np.int64),
        np.arange(3, dtype=np.int64),
        (2, 2),
    )
    mod = ModuleType("arrowspace")

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    adapter = _ArrowSpaceAdapter(mod, cache_size=1)
    adapter.build_index(DATASET_ID, np.zeros((2, 2), dtype=np.float64), tmp_path)
    return adapter, aspace


def test_search_without_k_passes_topk_contract(tmp_path):
    adapter, aspace = make_adapter(tmp_path)
    adapter.search(DATASET_ID, {"vector": [0.0, 0.0], "tau": 1.0})
    args, _ = aspace.search.call_args
    # (q_arr, gl, tau) — no k positional on the k-less path
    assert len(args) == 3
    assert args[2] == 1.0


def test_search_with_explicit_k_is_forwarded(tmp_path):
    adapter, aspace = make_adapter(tmp_path)
    adapter.search(DATASET_ID, {"vector": [0.0, 0.0], "tau": 1.0, "k": 1})
    args, _ = aspace.search.call_args
    assert len(args) == 4
    assert args[3] == 1


def test_search_default_tau(tmp_path):
    adapter, aspace = make_adapter(tmp_path)
    adapter.search(DATASET_ID, {"vector": [0.0, 0.0]})
    args, _ = aspace.search.call_args
    assert args[2] == 1.0


def test_typing_error_inside_search_propagates(tmp_path):
    """A real TypeError from search must propagate, not become a 501."""

    def bad_search(item, gl, tau, k):
        raise TypeError("k must be an int")

    adapter, _ = make_adapter(tmp_path, search_impl=bad_search)
    with pytest.raises(TypeError, match="k must be an int"):
        adapter.search(DATASET_ID, {"vector": [0.0, 0.0], "tau": 1.0, "k": 1})


def test_installed_arrowspace_supports_query_k():
    """Real library: the resolved ArrowSpace is the k-capable minimum."""
    installed = Version(importlib.metadata.version("arrowspace"))
    assert installed >= MIN_ARROWSPACE, (
        f"arrowspace {installed} installed; pyproject requires >=0.28.1"
    )


@pytest.mark.skipif(
    Version(importlib.metadata.version("arrowspace")) < MIN_ARROWSPACE,
    reason="needs arrowspace>=0.28.1 with query-time k",
)
def test_real_arrowspace_search_with_and_without_k(tmp_path):
    """End-to-end against the real ArrowSpace binding, not a fake builder."""
    rng = np.random.default_rng(42)
    items = rng.standard_normal((64, 8)).astype(np.float64)
    params = {"eps": 1.5, "k": 6, "topk": 5, "p": 2.0, "sigma": 1.0}
    aspace, gl = arrowspace.ArrowSpaceBuilder().build(params, items)

    adapter = _ArrowSpaceAdapter(arrowspace, cache_size=1)
    adapter._cache.put(
        DATASET_ID,
        _IndexEntry(aspace=aspace, gl=gl, nitems=aspace.nitems, nfeatures=8, nclusters=1),
    )

    bare = adapter.search(DATASET_ID, {"vector": items[0].tolist(), "tau": 1.0})
    capped = adapter.search(DATASET_ID, {"vector": items[0].tolist(), "tau": 1.0, "k": 3})

    assert bare["backend"] == "arrowspace"
    assert all("index" in r and "score" in r for r in bare["results"])
    assert len(capped["results"]) <= 3
