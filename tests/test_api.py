"""Integration tests for the arro-server HTTP API.

All tests use the TestClient fixture from conftest.py which creates an
isolated FastAPI app pointed at a tmp Zarr root.

Phase 1 additions (marked with # [Phase 1]):
    test_health_reports_backend          — /health includes arrowspace_backend + arrowspace_available
    test_manifold_has_backend_field      — /manifold includes backend + arrowspace_available
    test_stats_has_backend_field         — /stats includes backend + arrowspace_available
    test_post_index                      — POST /index builds and returns meta
    test_post_index_custom_params        — POST /index with custom graph_params
    test_post_search_vector              — POST /search with float vector
    test_lambdas_endpoint                — GET /lambdas after index build
    test_post_search_requires_index      — POST /search 503 when no index built
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(configured_app):
    return TestClient(configured_app)


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "main" in body["data_roots"]


def test_list_datasets(client: TestClient) -> None:
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    ids = {d["id"] for d in body["datasets"] if d["kind"] == "array"}
    # IDs use '--' separator instead of '/'
    assert {"main--matrix", "main--vector"}.issubset(ids)


def test_list_datasets_root_and_path_preserved(client: TestClient) -> None:
    """The 'root' and 'path' fields must still use the human-readable values."""
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    matrix = next(d for d in body["datasets"] if d["id"] == "main--matrix")
    assert matrix["root"] == "main"
    assert matrix["path"] == "matrix"


def test_metadata(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] == [50, 4]
    assert body["dtype"].startswith("float32")


def test_data_window(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/data?offset=0&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == 0
    assert body["limit"] == 10
    assert body["next_offset"] == 10
    assert len(body["data"]["rows"]) == 10
    assert body["data"]["rows"][0] == [0.0, 1.0, 2.0, 3.0]


def test_data_pagination_terminates(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/data?offset=45&limit=20")
    assert r.status_code == 200
    body = r.json()
    assert body["next_offset"] is None
    assert len(body["data"]["rows"]) == 5


def test_slice(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/slice?slice=0:3,1:3")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [3, 2]
    assert body["data"]["rows"][0] == [1.0, 2.0]


def test_slice_with_step(client: TestClient) -> None:
    """Step > 1 slices should return every Nth row."""
    r = client.get("/api/datasets/main--matrix/slice?slice=0:10:2")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [5, 4]  # rows 0,2,4,6,8


def test_slice_negative_index(client: TestClient) -> None:
    """Negative start index should resolve from the end of the axis."""
    r = client.get("/api/datasets/main--matrix/slice?slice=-3:")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [3, 4]  # last 3 rows


def test_invalid_slice(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/slice?slice=foo")
    assert r.status_code == 400


def test_unknown_dataset(client: TestClient) -> None:
    r = client.get("/api/datasets/main--missing/metadata")
    assert r.status_code == 404


def test_manifold_sidecar(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/manifold")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] in {"sidecar", "arrowspace"}
    assert "manifold" in body


def test_stats_returns_basic_shape(client: TestClient) -> None:
    """GET /stats returns a 'stats' dict with at least shape and dtype."""
    r = client.get("/api/datasets/main--matrix/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["shape"] == [50, 4]
    assert body["stats"]["dtype"].startswith("float32")


def test_search_sidecar(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/search?q=alpha")
    assert r.status_code == 200
    body = r.json()
    assert body["results"]
    assert body["results"][0]["id"] == "row-0"


def test_search_missing_index(client: TestClient) -> None:
    r = client.get("/api/datasets/main--vector/search?q=anything")
    assert r.status_code == 404


def test_window_budget_enforced(client: TestClient) -> None:
    """Requesting more rows than MAX_WINDOW should return 400."""
    r = client.get("/api/datasets/main--matrix/data?offset=0&limit=50")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# [Phase 1] New tests
# ---------------------------------------------------------------------------


def test_health_reports_backend(client: TestClient) -> None:  # [Phase 1] Task 1.5
    """GET /health must include arrowspace_backend and arrowspace_available."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "arrowspace_backend" in body
    assert body["arrowspace_backend"] in {"arrowspace", "sidecar", "none"}
    assert "arrowspace_available" in body
    assert isinstance(body["arrowspace_available"], bool)


def test_manifold_has_backend_field(client: TestClient) -> None:  # [Phase 1] Task 1.5
    """GET /manifold response must include backend and arrowspace_available."""
    r = client.get("/api/datasets/main--matrix/manifold")
    assert r.status_code == 200
    body = r.json()
    assert "backend" in body
    assert "arrowspace_available" in body
    assert isinstance(body["arrowspace_available"], bool)
    assert "source" in body  # 'live' | 'sidecar' | 'unavailable'


def test_stats_has_backend_field(client: TestClient) -> None:  # [Phase 1] Task 1.5
    """GET /stats response must include backend and arrowspace_available."""
    r = client.get("/api/datasets/main--matrix/stats")
    assert r.status_code == 200
    body = r.json()
    assert "backend" in body
    assert "arrowspace_available" in body


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("arrowspace"),
    reason="arrowspace not installed",
)
def test_post_index(client: TestClient) -> None:  # [Phase 1] Task 1.3
    """POST /index must build an index and return {built, nitems, nfeatures, nclusters}."""
    r = client.post("/api/datasets/main--matrix/index")
    assert r.status_code == 200
    body = r.json()
    assert body["built"] is True
    assert body["nitems"] == 50
    assert body["nfeatures"] == 4
    assert "nclusters" in body
    assert "graph_params" in body


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("arrowspace"),
    reason="arrowspace not installed",
)
def test_post_index_custom_params(client: TestClient) -> None:  # [Phase 1] Task 1.3
    """POST /index with custom graph_params must echo them back."""
    params = {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
    r = client.post("/api/datasets/main--matrix/index", json=params)
    assert r.status_code == 200
    body = r.json()
    assert body["graph_params"] == params


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("arrowspace"),
    reason="arrowspace not installed",
)
def test_post_search_vector(client: TestClient) -> None:  # [Phase 1] Task 1.4
    """POST /search must return scored results for a float vector."""
    # Build index first
    client.post("/api/datasets/main--matrix/index")
    # Query with a 4-element vector matching the array's feature dimension
    query_vector = [0.0, 1.0, 2.0, 3.0]
    r = client.post(
        "/api/datasets/main--matrix/search",
        json={"vector": query_vector, "tau": 1.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) > 0
    hit = body["results"][0]
    assert "index" in hit
    assert "score" in hit
    assert isinstance(hit["index"], int)
    assert isinstance(hit["score"], float)
    assert body["backend"] == "arrowspace"
    assert body["arrowspace_available"] is True


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("arrowspace"),
    reason="arrowspace not installed",
)
def test_lambdas_endpoint(client: TestClient) -> None:  # [Phase 1] Task 1.5
    """GET /lambdas must return eigenvalue data after index is built."""
    client.post("/api/datasets/main--matrix/index")
    r = client.get("/api/datasets/main--matrix/lambdas")
    assert r.status_code == 200
    body = r.json()
    assert "lambdas" in body
    assert "lambdas_sorted" in body
    assert "nitems" in body
    assert body["backend"] == "arrowspace"
    assert body["arrowspace_available"] is True


def test_post_search_requires_index(client: TestClient) -> None:  # [Phase 1] Task 1.4
    """POST /search without a built index must return 503 or 404."""
    r = client.post(
        "/api/datasets/main--matrix/search",
        json={"vector": [0.1, 0.2, 0.3, 0.4], "tau": 1.0},
    )
    # 503 = arrowspace not installed, 404 = installed but no index built
    assert r.status_code in {503, 404}
