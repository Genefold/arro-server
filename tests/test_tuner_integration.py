"""tests/test_tuner_integration.py — smoke tests for arrowspace-tuner."""

import importlib.metadata

import arrowspace_tuner as at
import numpy as np
from packaging.version import Version


def test_tune_import():
    assert hasattr(at, "tune")


def test_tune_return_shape():
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((200, 32)).astype(np.float64)
    params = at.tune(embeddings, n_trials=5, sample_n=50)
    assert set(params.keys()) == {"eps", "k", "topk", "p", "sigma"}
    assert isinstance(params["eps"], float)
    assert isinstance(params["k"], int)
    assert isinstance(params["topk"], int)
    assert isinstance(params["p"], float)
    assert isinstance(params["sigma"], (float, type(None)))


def test_tune_version_in_range():
    """Allow patch bumps within the declared >=0.5.0,<0.6 range."""
    v = Version(importlib.metadata.version("arrowspace-tuner"))
    assert Version("0.5.0") <= v < Version("0.6.0")


def test_tuner_params_accepted_by_index_route(tmp_path):
    """Tuner output feeds POST /api/datasets/{id}/index unchanged (lossless by design)."""
    import os
    import sys
    import types
    from unittest.mock import MagicMock

    import zarr
    from fastapi.testclient import TestClient

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    n = 10
    aspace = MagicMock()
    aspace.nitems = n
    aspace.nfeatures = 32
    aspace.nclusters = 2
    aspace.lambdas_sorted.return_value = [(float(n - i), i) for i in range(n)]
    gl = MagicMock()
    gl.nnodes = n
    gl.shape = (n, n)
    gl.to_csr.return_value = (
        np.ones(n, dtype=np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64),
        (n, n),
    )

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod = types.ModuleType("arrowspace")
    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]

    root_dir = tmp_path / "data"
    root_dir.mkdir()
    zarr_dir = root_dir / "matrix"
    arr = zarr.open(str(zarr_dir), mode="w", shape=(50, 32), chunks=(10, 32), dtype="float32")
    arr[:] = np.arange(50 * 32, dtype="float32").reshape(50, 32)
    index_store = tmp_path / "index_store"
    index_store.mkdir()

    rng = np.random.default_rng(0)
    tuner_params = at.tune(
        rng.standard_normal((200, 32)).astype(np.float64), n_trials=3, sample_n=50
    )

    sys.modules["arrowspace"] = fake_mod  # type: ignore[assignment]
    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={root_dir}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(index_store)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()
    try:
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/datasets/main--matrix/index",
                json=tuner_params,
            )
            assert resp.status_code == 200, resp.text
            # Route echoes effective graph_params; tuner keys must survive transit.
            assert resp.json()["graph_params"] == tuner_params
    finally:
        sys.modules.pop("arrowspace", None)
        os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
        os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
        os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
        settings_mod.reset_settings_cache()
        registry_mod.get_registry.cache_clear()
        reset_adapter_cache()