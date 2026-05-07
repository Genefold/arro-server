from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(configured_app):
    return TestClient(configured_app)


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
    assert {"main/matrix", "main/vector"}.issubset(ids)


def test_metadata(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] == [50, 4]
    assert body["dtype"].startswith("float32")


def test_data_window(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/data?offset=0&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == 0
    assert body["limit"] == 10
    assert body["next_offset"] == 10
    assert len(body["data"]["rows"]) == 10
    assert body["data"]["rows"][0] == [0.0, 1.0, 2.0, 3.0]


def test_data_pagination_terminates(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/data?offset=45&limit=20")
    assert r.status_code == 200
    body = r.json()
    assert body["next_offset"] is None
    assert len(body["data"]["rows"]) == 5


def test_slice(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/slice?slice=0:3,1:3")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [3, 2]
    assert body["data"]["rows"][0] == [1.0, 2.0]


def test_invalid_slice(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/slice?slice=foo")
    assert r.status_code == 400


def test_unknown_dataset(client: TestClient) -> None:
    r = client.get("/api/datasets/main/missing/metadata")
    assert r.status_code == 404


def test_manifold_sidecar(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/manifold")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] in {"sidecar", "pyarrowspace"}
    assert "manifold" in body


def test_stats_combines_basic_and_arrowspace(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["basic"]["shape"] == [50, 4]
    assert "arrowspace" in body


def test_search_sidecar(client: TestClient) -> None:
    r = client.get("/api/datasets/main/matrix/search?q=alpha")
    assert r.status_code == 200
    body = r.json()
    assert body["results"]
    assert body["results"][0]["id"] == "row-0"


def test_search_missing_index(client: TestClient) -> None:
    r = client.get("/api/datasets/main/vector/search?q=anything")
    assert r.status_code == 404
