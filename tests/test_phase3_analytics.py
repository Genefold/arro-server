"""Phase 3 analytics endpoint tests.

Uses the shared conftest's tmp_zarr_root to create a real Zarr dataset,
then injects a pre-configured _ArrowSpaceAdapter into the app via
dependency_overrides so tests run without the arrowspace package.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arro_server.api.routes import _arrowspace
from arro_server.app import create_app
from arro_server.arrowspace_adapter import (
    _ArrowSpaceAdapter,
    _IndexEntry,
    reset_adapter_cache,
)
from arro_server.settings import reset_settings_cache
from arro_server.storage.registry import reset_registry_cache

N_ITEMS = 20
N_FEATURES = 8
RNG = np.random.default_rng(42)

# We reuse "main--matrix" from conftest's tmp_zarr_root so reg.open() works.
DATASET_ID = "main--matrix"


def _make_mock_aspace() -> MagicMock:
    m = MagicMock()
    m.nitems = N_ITEMS
    m.nfeatures = N_FEATURES
    m.nclusters = 4

    lambdas = np.sort(np.abs(RNG.standard_normal(N_ITEMS))).astype(np.float64)
    lambdas[0] = 0.0

    m.lambdas.return_value = lambdas
    m.lambdas_sorted.return_value = sorted(
        [(float(v), int(i)) for i, v in enumerate(lambdas)],
        key=lambda x: x[0],
    )
    m.spot_motives_eigen.return_value = [(i, float(i * 0.1)) for i in range(5)]
    m.spot_motives_energy.return_value = [(i, float(i * 0.2)) for i in range(5)]
    m.spot_subg_centroids.return_value = [(i, float(i * 0.3)) for i in range(3)]
    m.spot_subg_motives.return_value = [(i, float(i * 0.4)) for i in range(3)]
    m.search.return_value = [(i, float(i * 0.1)) for i in range(5)]
    m.search_batch.return_value = [
        [(i, float(i * 0.1)) for i in range(5)],
        [(i, float(i * 0.15)) for i in range(5)],
    ]
    m.search_energy.return_value = [(i, float(i * 0.1)) for i in range(5)]
    m.search_hybrid.return_value = [(i, float(i * 0.1)) for i in range(5)]
    m.search_linear_sorted.return_value = [(i, float(i * 0.1)) for i in range(5)]
    return m


def _make_mock_gl() -> MagicMock:
    dense = RNG.standard_normal((N_ITEMS, N_ITEMS)).astype(np.float32)
    m = MagicMock()
    m.nnodes = N_ITEMS
    m.shape = (N_ITEMS, N_ITEMS)
    m.graph_params = {"eps": 1.2, "k": 5, "topk": 3, "p": 2.0, "sigma": None}

    data = dense.flatten()
    indices = np.tile(np.arange(N_ITEMS, dtype=np.int64), N_ITEMS)
    indptr = np.arange(0, N_ITEMS * N_ITEMS + 1, N_ITEMS, dtype=np.int64)
    m.to_csr.return_value = (data, indices, indptr, (N_ITEMS, N_ITEMS))
    m.to_dense.return_value = dense
    return m


@pytest.fixture
def tmp_index_store(tmp_path: Path) -> Path:
    d = tmp_path / "index_store"
    d.mkdir()
    return d


@pytest.fixture
def client(tmp_zarr_root: Path, tmp_index_store: Path) -> TestClient:
    mock_mod = MagicMock()
    adapter = _ArrowSpaceAdapter(mock_mod, cache_size=4)
    entry = _IndexEntry(
        aspace=_make_mock_aspace(),
        gl=_make_mock_gl(),
        nitems=N_ITEMS,
        nfeatures=N_FEATURES,
        nclusters=4,
    )
    adapter._cache.put(DATASET_ID, entry)

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    os.environ["ARRO_SERVER_INDEX_STORE"] = str(tmp_index_store)
    reset_settings_cache()
    reset_registry_cache()
    reset_adapter_cache()

    app = create_app()
    app.dependency_overrides[_arrowspace] = lambda: adapter

    with TestClient(app) as c:
        yield c

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    os.environ.pop("ARRO_SERVER_INDEX_STORE", None)
    reset_settings_cache()
    reset_registry_cache()
    reset_adapter_cache()


# ===========================================================================
# Tests
# ===========================================================================


class TestMotives:
    def test_motives_eigen(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/motives?mode=eigen")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "eigen"
        assert isinstance(body["motives"], list)
        assert body["count"] == len(body["motives"])
        assert all("index" in m and "score" in m for m in body["motives"])

    def test_motives_energy(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/motives?mode=energy")
        assert r.status_code == 200
        assert r.json()["mode"] == "energy"

    def test_motives_invalid_mode(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/motives?mode=invalid")
        assert r.status_code == 422


class TestSubgraphs:
    def test_subgraphs_motives(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/subgraphs?mode=motives")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "motives"
        assert isinstance(body["subgraphs"], list)

    def test_subgraphs_centroids(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/subgraphs?mode=centroids")
        assert r.status_code == 200
        assert r.json()["mode"] == "centroids"


class TestGraphExport:
    def test_graph_csr(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/graph?fmt=csr")
        assert r.status_code == 200
        body = r.json()
        assert body["fmt"] == "csr"
        assert "data" in body
        assert "indices" in body
        assert "indptr" in body
        assert len(body["shape"]) == 2

    def test_graph_dense(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/graph?fmt=dense")
        assert r.status_code == 200
        body = r.json()
        assert body["fmt"] == "dense"
        assert isinstance(body["matrix"], list)
        assert body["nnodes"] == N_ITEMS

    def test_graph_invalid_fmt(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/graph?fmt=xml")
        assert r.status_code == 422


class TestLambdas:
    def test_lambdas(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/lambdas")
        assert r.status_code == 200
        body = r.json()
        assert "lambdas" in body
        assert "lambdas_sorted" in body
        assert body["nitems"] == N_ITEMS


class TestSpectralMetrics:
    def test_spectral_metrics(self, client: TestClient) -> None:
        r = client.get(f"/api/datasets/{DATASET_ID}/spectral_metrics")
        assert r.status_code == 200
        body = r.json()
        required_keys = [
            "fiedler_value",
            "spectral_gap",
            "spectral_energy_total",
            "spectral_energy_norm",
            "lambda_percentiles",
            "lambdas_sorted",
            "algebraic_connectivity",
            "lambda_min",
            "lambda_max",
        ]
        for key in required_keys:
            assert key in body, f"Missing key: {key}"
        assert 0.0 <= body["spectral_energy_norm"] <= 1.0 + 1e-9
        assert isinstance(body["lambda_percentiles"], dict)
        assert "p50" in body["lambda_percentiles"]


class TestSearchMode:
    def test_search_mode_taumode(self, client: TestClient) -> None:
        vec = RNG.standard_normal(N_FEATURES).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/mode",
            json={"vector": vec, "mode": "taumode", "tau": 1.0},
        )
        assert r.status_code == 200
        assert r.json()["mode"] == "taumode"

    def test_search_mode_hybrid(self, client: TestClient) -> None:
        vec = RNG.standard_normal(N_FEATURES).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/mode",
            json={"vector": vec, "mode": "hybrid", "alpha": 0.6},
        )
        assert r.status_code == 200
        assert r.json()["mode"] == "hybrid"

    def test_search_mode_energy(self, client: TestClient) -> None:
        vec = RNG.standard_normal(N_FEATURES).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/mode",
            json={"vector": vec, "mode": "energy", "k": 5},
        )
        assert r.status_code == 200

    def test_search_mode_linear_sorted(self, client: TestClient) -> None:
        vec = RNG.standard_normal(N_FEATURES).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/mode",
            json={"vector": vec, "mode": "linear_sorted", "k": 5},
        )
        assert r.status_code == 200

    def test_search_mode_invalid(self, client: TestClient) -> None:
        vec = RNG.standard_normal(N_FEATURES).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/mode",
            json={"vector": vec, "mode": "cosine"},
        )
        assert r.status_code == 422


class TestSearchBatch:
    def test_search_batch(self, client: TestClient) -> None:
        vecs = RNG.standard_normal((3, N_FEATURES)).tolist()
        r = client.post(
            f"/api/datasets/{DATASET_ID}/search/batch",
            json={"vectors": vecs, "tau": 1.0},
        )
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert len(body["results"]) == 2
