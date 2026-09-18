"""Regression tests for the F2 bug found by the E2E smoke test.

_routes._arrowspace() called load_arrowspace() with no arguments, so the
lru-cached adapter was built with tune_store=None and tuned params never
reached build_index — silent fallback to DEFAULT_GRAPH_PARAMS. The route
also pre-injected DEFAULT_GRAPH_PARAMS when no body was sent, defeating
_resolve_graph_params' user > tuned > default chain.

Two levels of coverage:
- unit: the dependency must thread app.state.tune_store into load()
- HTTP: no-body POST /index must forward tuned params to the builder

Both tests are environment-independent: no real arrowspace import is
required (fake module injection / load() monkeypatching).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arro_server.arrowspace_adapter import reset_adapter_cache
from arro_server.storage.tune_store import TunedParams, TuneStore

TUNED = TunedParams(
    eps=0.821,
    k=10,
    topk=5,
    p=2.0,
    sigma=0.5,
    score=2.137,  # real tuner output — unbounded objective
    tuned_at="2026-09-18T10:18:01+00:00",
    dataset="main--matrix",
)


def make_fake_arrowspace_module() -> ModuleType:
    """Fake arrowspace module whose builder captures graph_params.

    Mirrors tests/test_tuned_param_resolution.py:make_fake_module so the
    builder call can be inspected without the real Rust package.
    """
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

    class FakeBuilder:
        def build(self, graph_params, array):
            captured["graph_params"] = graph_params
            return aspace, gl

    mod = ModuleType("arrowspace")
    mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    mod.captured = captured  # type: ignore[attr-defined]
    return mod


class TestArrowspaceDependencyWiring:
    def test_dependency_threads_tune_store_from_app_state(self, monkeypatch):
        """_arrowspace must pass app.state.tune_store to load_arrowspace."""
        captured: dict = {}

        def fake_load(tune_store=None):
            captured["tune_store"] = tune_store
            return MagicMock()

        monkeypatch.setattr("arro_server.api.routes.load_arrowspace", fake_load)

        store = TuneStore(Path("/tmp/unused-tune.json"))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(tune_store=store)))

        from arro_server.api.routes import _arrowspace

        _arrowspace(request)
        assert captured["tune_store"] is store


@pytest.fixture
def wired_app(tmp_zarr_root: Path, tmp_path: Path, monkeypatch):
    """Real create_app with lifespan; fake arrowspace; isolated stores."""
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    monkeypatch.setitem(sys.modules, "arrowspace", make_fake_arrowspace_module())
    monkeypatch.setenv("ARRO_SERVER_DATA_ROOTS", f"main={tmp_zarr_root}")
    monkeypatch.setenv("ARRO_SERVER_SERVE_FRONTEND", "false")
    monkeypatch.setenv("ARRO_SERVER_TUNE_PARAMS_PATH", str(tmp_path / "tune_params.json"))
    monkeypatch.setenv("ARRO_SERVER_INDEX_STORE", str(tmp_path / "index_store"))
    settings_mod.reset_settings_cache()
    # Full singleton reset: reset_registry_cache() only invalidates the cached
    # dataset dict, but the singleton keeps the roots baked in at creation
    # time. A singleton created by an earlier test (with that test's tmp_path)
    # would resolve main--matrix against the wrong root and 404. CI's file
    # ordering exposed this; conftest.configured_app does the same reset.
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    yield create_app()

    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


class TestHttpBuildUsesTunedParams:
    def test_no_body_index_build_forwards_tuned_params_to_builder(self, wired_app):
        fake_mod = sys.modules["arrowspace"]
        with TestClient(wired_app) as client:
            store: TuneStore = wired_app.state.tune_store
            assert store is not None
            store.set("main--matrix", TUNED)

            resp = client.post("/api/datasets/main--matrix/index")

        assert resp.status_code == 200, resp.text
        gp = resp.json()["graph_params"]
        assert gp["eps"] == pytest.approx(0.821)
        assert gp["k"] == 10
        assert gp["topk"] == 5
        # The tuned params must reach the actual builder call, not just the response.
        assert fake_mod.captured["graph_params"] == gp

    def test_explicit_body_params_still_win_over_tuned(self, wired_app):
        with TestClient(wired_app) as client:
            store: TuneStore = wired_app.state.tune_store
            store.set("main--matrix", TUNED)

            resp = client.post(
                "/api/datasets/main--matrix/index",
                json={"eps": 2.0},
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["graph_params"] == {"eps": 2.0}

    def test_no_tuned_entry_falls_back_to_defaults(self, wired_app):
        with TestClient(wired_app) as client:
            resp = client.post("/api/datasets/main--matrix/index")

        assert resp.status_code == 200, resp.text
        gp = resp.json()["graph_params"]
        assert gp["eps"] == 1.31
        assert gp["k"] == 30
