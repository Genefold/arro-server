"""taumode search must forward the request's k to the library.

Library contract: search(item, gl, tau, k=None) on arrowspace>=0.28.1;
0.26.x has no k parameter. The adapter forwards explicit k where supported
and returns 501 otherwise; k absent keeps the index's topk on every version.
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import HTTPException

from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

DATASET_ID = "test/ds"


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


def test_search_explicit_k_on_old_library_returns_501(tmp_path):
    """arrowspace 0.26.x search(item, gl, tau) — explicit k must 501, not 500."""

    def old_search(item, gl, tau):
        return [(0, 1.0)]

    adapter, _ = make_adapter(tmp_path, search_impl=old_search)
    with pytest.raises(HTTPException) as exc_info:
        adapter.search(DATASET_ID, {"vector": [0.0, 0.0], "tau": 1.0, "k": 1})
    assert exc_info.value.status_code == 501
    assert "0.28.1" in exc_info.value.detail
