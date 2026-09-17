"""Tests for Issue #68 — tuned param resolution in build_index.

Priority chain: user_params > TuneStore > DEFAULT_GRAPH_PARAMS.
TuneStore is a MagicMock — no real persistence needed.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arro_server.arrowspace_adapter import (
    DEFAULT_GRAPH_PARAMS,
    MANIFEST_FILENAME,
    _ArrowSpaceAdapter,
)
from arro_server.storage.tune_store import TunedParams

DATASET_ID = "test/ds"


def make_tuned(**overrides) -> TunedParams:
    defaults = dict(
        eps=1.1,
        k=20,
        topk=15,
        p=1.9,
        sigma=0.7,
        score=0.9,
        tuned_at="2026-09-17T14:00:00+00:00",
        dataset=DATASET_ID,
    )
    return TunedParams(**(defaults | overrides))


def make_fake_module() -> types.ModuleType:
    """Fake arrowspace module whose builder captures graph_params."""
    aspace = MagicMock()
    aspace.nitems = 4
    aspace.nfeatures = 3
    aspace.nclusters = 1
    aspace.lambdas.return_value = []
    aspace.lambdas_sorted.return_value = []
    gl = MagicMock()
    gl.nnodes = 4
    gl.to_csr.return_value = (
        np.array([1.0], dtype=np.float32),
        np.array([0], dtype=np.int64),
        np.arange(5, dtype=np.int64),
        (4, 4),
    )
    captured: dict = {}
    captured["gl"] = gl

    class FakeBuilder:
        def build(self, graph_params, array):
            captured["graph_params"] = graph_params
            return aspace, gl

    mod = types.ModuleType("arrowspace")
    mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    mod.captured = captured  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def fake_mod() -> types.ModuleType:
    return make_fake_module()


# ---------------------------------------------------------------------------
# _resolve_graph_params — the three-case chain
# ---------------------------------------------------------------------------


class TestResolveGraphParams:
    def _adapter(self, tune_store=None, fake_mod=None) -> _ArrowSpaceAdapter:
        return _ArrowSpaceAdapter(fake_mod or object(), cache_size=1, tune_store=tune_store)

    def test_user_params_win_over_tuned(self, fake_mod):
        store = MagicMock()
        store.get.return_value = make_tuned()
        user = {"eps": 2.0}
        gp, source = self._adapter(store, fake_mod)._resolve_graph_params(DATASET_ID, user)
        assert gp is user
        assert source == "user"

    def test_tuned_fallback(self, fake_mod):
        store = MagicMock()
        store.get.return_value = make_tuned()
        gp, source = self._adapter(store, fake_mod)._resolve_graph_params(DATASET_ID, None)
        assert gp == {"eps": 1.1, "k": 20, "topk": 15, "p": 1.9, "sigma": 0.7}
        assert source == "tuned"

    def test_default_fallback_when_no_tuned_entry(self, fake_mod):
        store = MagicMock()
        store.get.return_value = None
        gp, source = self._adapter(store, fake_mod)._resolve_graph_params(DATASET_ID, None)
        assert gp is DEFAULT_GRAPH_PARAMS
        assert source == "default"

    def test_default_when_no_store_injected(self, fake_mod):
        gp, source = self._adapter(None, fake_mod)._resolve_graph_params(DATASET_ID, None)
        assert gp is DEFAULT_GRAPH_PARAMS
        assert source == "default"

    def test_partial_user_params_not_merged(self, fake_mod):
        store = MagicMock()
        store.get.return_value = make_tuned()
        user = {"eps": 2.0}
        gp, source = self._adapter(store, fake_mod)._resolve_graph_params(DATASET_ID, user)
        assert gp == {"eps": 2.0}
        assert "k" not in gp
        assert source == "user"

    def test_tuned_sigma_none_falls_back_to_default(self, fake_mod):
        store = MagicMock()
        store.get.return_value = make_tuned(sigma=None)
        gp, source = self._adapter(store, fake_mod)._resolve_graph_params(DATASET_ID, None)
        assert gp["sigma"] == DEFAULT_GRAPH_PARAMS["sigma"]
        assert source == "tuned"


# ---------------------------------------------------------------------------
# build_index integration: resolved params flow into builder + manifest
# ---------------------------------------------------------------------------


class TestBuildIndexResolution:
    def test_build_uses_tuned_params(self, fake_mod, tmp_path: Path):
        store = MagicMock()
        store.get.return_value = make_tuned()
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=1, tune_store=store)
        adapter.build_index(DATASET_ID, np.zeros((4, 3)), tmp_path)
        builder_params = fake_mod.captured["graph_params"]
        assert builder_params == {"eps": 1.1, "k": 20, "topk": 15, "p": 1.9, "sigma": 0.7}

    def test_manifest_records_resolved_params(self, fake_mod, tmp_path: Path):
        store = MagicMock()
        store.get.return_value = make_tuned()
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=1, tune_store=store)
        adapter.build_index(DATASET_ID, np.zeros((4, 3)), tmp_path)
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert manifest[DATASET_ID]["graph_params"] == {
            "eps": 1.1,
            "k": 20,
            "topk": 15,
            "p": 1.9,
            "sigma": 0.7,
        }

    def test_manifest_records_user_params_as_given(self, fake_mod, tmp_path: Path):
        store = MagicMock()
        store.get.return_value = make_tuned()
        adapter = _ArrowSpaceAdapter(fake_mod, cache_size=1, tune_store=store)
        user = {"eps": 2.0}
        adapter.build_index(DATASET_ID, np.zeros((4, 3)), tmp_path, graph_params=user)
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert manifest[DATASET_ID]["graph_params"] == user
