"""Tests for POST /api/prompts/nl_search, POST /api/prompts/search,
GET /api/prompts/health, and GET /api/prompts/warm.

All heavy dependencies (PromptSearchEngine, EmbedderService) are replaced
with lightweight stubs so tests run without GPU, model weights, or data files.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

FAKE_DIM   = 768
FAKE_QUERY = "write a DALL-E prompt for minimalist art"


def _fake_record(i: int) -> dict[str, Any]:
    return {
        "id":               f"pk_{i:05d}",
        "title":            f"Prompt title {i}",
        "body":             f"Prompt body {i}",
        "tags":             ["art", "dalle"],
        "upvotes":          i * 3,
        "views":            i * 100,
        "author_reputation": float(i),
        "_score":           round(0.9 - i * 0.05, 6),
        "_salience":        round(0.8 - i * 0.05, 6),
        "_tau":             0.75,
    }


def _make_stub_engine(n: int = 5):
    """Return a MagicMock that quacks like PromptSearchEngine."""
    engine = MagicMock()
    engine.aspace.nitems    = 1000
    engine.aspace.nfeatures = FAKE_DIM
    engine.aspace.nclusters = 20
    engine.gl.nnodes        = 1000
    engine.ids              = [f"pk_{i:05d}" for i in range(n)]
    engine.embs             = np.random.rand(n, FAKE_DIM)
    engine.search.return_value = [_fake_record(i) for i in range(n)]
    return engine


def _make_stub_embedder():
    """Return a MagicMock that quacks like EmbedderService."""
    embedder = MagicMock()
    embedder.model_name = "nomic-ai/nomic-embed-text-v1.5"
    embedder.dim        = FAKE_DIM
    embedder.embed.return_value = np.random.rand(FAKE_DIM).astype(np.float64)
    return embedder


# ---------------------------------------------------------------------------
# App fixture with stubs injected
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_app():
    """
    Creates the FastAPI app with PromptSearchEngine and EmbedderService
    replaced by stubs.  No data files or GPU required.
    """
    from arro_server import settings as settings_mod
    from arro_server import arrowspace_adapter
    from arro_server.app import create_app
    from arro_server import storage as storage_mod
    from arro_server.search_engine import PromptSearchEngine

    # env
    os.environ["ARRO_SERVER_DATA_ROOTS"]       = ""
    os.environ["ARRO_SERVER_SERVE_FRONTEND"]   = "false"
    os.environ["ARRO_SERVER_PROMPT_DATA_DIR"]  = "/nonexistent"

    settings_mod.reset_settings_cache()
    storage_mod.registry.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()
    PromptSearchEngine.reset()

    stub_engine   = _make_stub_engine()
    stub_embedder = _make_stub_embedder()

    # Patch the singletons that the route helpers call
    with (
        patch("arro_server.api.routes.PromptSearchEngine") as MockEngine,
        patch("arro_server.embedder.EmbedderService")      as MockEmbedder,
    ):
        # _get_engine() calls PromptSearchEngine.get() -> return stub
        MockEngine.get.return_value             = stub_engine
        MockEngine._instance                    = stub_engine
        # _get_embedder() does `from ..embedder import EmbedderService; EmbedderService.get()`
        MockEmbedder.get.return_value           = stub_embedder
        MockEmbedder._instance                  = stub_embedder

        app = create_app()
        yield app

    # teardown
    PromptSearchEngine.reset()
    os.environ.pop("ARRO_SERVER_DATA_ROOTS",      None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND",   None)
    os.environ.pop("ARRO_SERVER_PROMPT_DATA_DIR",  None)
    settings_mod.reset_settings_cache()
    storage_mod.registry.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture
def client(stub_app):
    return TestClient(stub_app)


# ---------------------------------------------------------------------------
# GET /api/prompts/health
# ---------------------------------------------------------------------------

class TestPromptsHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/api/prompts/health")
        assert r.status_code == 200

    def test_schema(self, client: TestClient):
        data = client.get("/api/prompts/health").json()
        assert "status" in data
        assert "prompt_engine_ready" in data
        assert "embedder_ready" in data


# ---------------------------------------------------------------------------
# POST /api/prompts/nl_search
# ---------------------------------------------------------------------------

class TestNLSearch:
    def test_valid_query_returns_200(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 5})
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 5}
        ).json()
        assert data["query"] == FAKE_QUERY
        assert data["k"] == 5
        assert isinstance(data["results"], list)
        assert "result_count" in data
        assert "tau" in data
        assert "lam" in data

    def test_result_fields(self, client: TestClient):
        results = client.post(
            "/api/prompts/nl_search", json={"query": FAKE_QUERY}
        ).json()["results"]
        assert len(results) > 0
        first = results[0]
        assert "id" in first
        assert "_score" in first
        assert "_salience" in first
        assert "_tau" in first

    def test_empty_query_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": ""})
        assert r.status_code == 422

    def test_missing_query_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"k": 5})
        assert r.status_code == 422

    def test_k_out_of_range_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 0})
        assert r.status_code == 422

    def test_tau_propagated(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "k": 3, "tau": 1.5},
        ).json()
        assert data["tau"] == 1.5

    def test_lam_propagated(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "lam": 0.5},
        ).json()
        assert data["lam"] == 0.5


# ---------------------------------------------------------------------------
# POST /api/prompts/search  (pre-embedded vector)
# ---------------------------------------------------------------------------

class TestPromptVectorSearch:
    def _vec(self) -> list[float]:
        return list(np.random.rand(FAKE_DIM).astype(float))

    def test_valid_vector_returns_200(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"vector": self._vec(), "k": 5})
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        data = client.post(
            "/api/prompts/search", json={"vector": self._vec(), "k": 5}
        ).json()
        assert "results" in data
        assert "result_count" in data
        assert data["query"] is None  # no NL query for this endpoint

    def test_wrong_dim_422(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"vector": [0.1, 0.2, 0.3], "k": 5})
        assert r.status_code == 422

    def test_missing_vector_422(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"k": 5})
        assert r.status_code == 422
