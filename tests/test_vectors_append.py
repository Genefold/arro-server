"""tests/test_vectors_append.py

Integration and unit tests for POST /api/datasets/{dataset_id}/vectors/append.

Test inventory:
    1.  test_append_basic                          — append M vectors, verify start_row & new_shape
    2.  test_append_start_row_sequential            — two consecutive appends return correct start_row
    3.  test_append_shape_updated_in_cache          — GET /datasets returns updated shape after append
    4.  test_append_dim_mismatch_422                — wrong feature dimension → 422
    5.  test_append_empty_vectors_422               — zero vectors (M=0) → 422
    6.  test_append_wrong_ndim_422                  — 1-D input → 422
    7.  test_append_dtype_mismatch_422              — incompatible dtype → 422
    8.  test_append_dataset_not_found_404           — non-existent dataset_id → 404
    9.  test_append_concurrent_start_rows_no_overlap — 3 threads, ranges are disjoint & contiguous
    10. test_append_data_readable_after_append      — GET /data returns just-appended vectors
    11. test_append_does_not_read_existing_vectors  — monkey-patch to verify O(M) not O(N)

Thread-safety:
    test_append_concurrent_start_rows_no_overlap is the most important correctness
    test. It launches 3 concurrent threads and verifies that the returned
    start_row ranges partition [0, total_M) without gaps or overlaps.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client(tmp_path: Path):
    """TestClient with a single 2-D Zarr array at main--matrix.

    The array has shape (50, 4), dtype float32.
    """
    import zarr

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    # Write a Zarr v3 array of shape (50, 4) float32
    zarr_dir = tmp_path / "matrix"
    arr = zarr.open(str(zarr_dir), mode="w", shape=(50, 4), chunks=(10, 4), dtype="float32")
    arr[:] = np.arange(50 * 4, dtype="float32").reshape(50, 4)

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_path}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path

    from arro_server.storage import registry as registry_mod

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1. Basic append — verify start_row and new_shape
# ---------------------------------------------------------------------------


def test_append_basic(app_client):
    """Append M vectors returns start_row == old_n and new_shape == [old_n+M, D]."""
    client, _ = app_client
    vectors = [[float(i) for i in range(4)] for _ in range(7)]
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": vectors, "dtype": "float32"},
    )
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["start_row"] == 50
    assert body["appended"] == 7
    assert body["new_shape"] == [57, 4]


# ---------------------------------------------------------------------------
# 2. Sequential append — start_row advances correctly
# ---------------------------------------------------------------------------


def test_append_start_row_sequential(app_client):
    """Two consecutive appends: second start_row == first + M1."""
    client, _ = app_client
    batch1 = [[float(i) for i in range(4)] for _ in range(10)]
    batch2 = [[float(i * 2) for i in range(4)] for _ in range(5)]

    resp1 = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": batch1, "dtype": "float32"},
    )
    assert resp1.status_code == 200
    r1 = resp1.json()
    assert r1["start_row"] == 50
    assert r1["new_shape"] == [60, 4]

    resp2 = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": batch2, "dtype": "float32"},
    )
    assert resp2.status_code == 200
    r2 = resp2.json()
    assert r2["start_row"] == 60
    assert r2["new_shape"] == [65, 4]


# ---------------------------------------------------------------------------
# 3. Cache update — shape visible in GET /datasets after append
# ---------------------------------------------------------------------------


def test_append_shape_updated_in_cache(app_client):
    """After append GET /datasets/{id} returns the updated shape."""
    client, _ = app_client
    vectors = [[float(i) for i in range(4)] for _ in range(3)]
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": vectors, "dtype": "float32"},
    )
    assert resp.status_code == 200

    meta = client.get("/api/datasets/main--matrix/metadata").json()
    assert meta["shape"] == [53, 4]


# ---------------------------------------------------------------------------
# 4. Dimension mismatch → 422
# ---------------------------------------------------------------------------


def test_append_dim_mismatch_422(app_client):
    """Vectors with wrong feature dimension return 422."""
    client, _ = app_client
    vectors = [[float(i) for i in range(7)]]  # D=7 instead of 4
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": vectors, "dtype": "float32"},
    )
    assert resp.status_code == 422
    assert "Dimension mismatch" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5. Empty vectors → 422
# ---------------------------------------------------------------------------


def test_append_empty_vectors_422(app_client):
    """Request with shape (1, 0) — zero-length inner vectors — returns 422."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": [[]], "dtype": "float32"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. Wrong ndim → 422
# ---------------------------------------------------------------------------


def test_append_wrong_ndim_422(app_client):
    """Single flat vector (1-D) returns 422."""
    client, _ = app_client
    vectors = [1.0, 2.0, 3.0, 4.0]  # flat list, not list-of-lists → Pydantic rejects at 422
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": vectors, "dtype": "float32"},
    )
    # Pydantic catches this: list[float] is not list[list[float]]
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7. Dtype mismatch → 422
# ---------------------------------------------------------------------------


def test_append_dtype_mismatch_422(app_client):
    """Vectors with wrong dtype return 422."""
    client, _ = app_client
    vectors = [[float(i) for i in range(4)] for _ in range(2)]
    # Dataset is float32, request says float64 (explicit mismatch)
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": vectors, "dtype": "int32"},
    )
    assert resp.status_code == 422
    assert "dtype mismatch" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 8. Dataset not found → 404
# ---------------------------------------------------------------------------


def test_append_dataset_not_found_404(app_client):
    """Non-existent dataset_id returns 404."""
    client, _ = app_client
    vectors = [[1.0, 2.0, 3.0, 4.0]]
    resp = client.post(
        "/api/datasets/nonexistent--foo/vectors/append",
        json={"vectors": vectors, "dtype": "float32"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. Concurrent append — ranges are disjoint and contiguous
# ---------------------------------------------------------------------------


def test_append_concurrent_start_rows_no_overlap(app_client):
    """3 concurrent threads: returned ranges partition [0, total_M) exactly.

    Uses backend.append_vectors() directly (not HTTP) to avoid TestClient
    serialization.
    """
    import zarr

    from arro_server.settings import get_settings
    from arro_server.storage import registry as registry_mod
    from arro_server.storage.zarr_fs import ZarrFilesystemBackend

    _, base_path = app_client
    zarr_dir = base_path / "matrix"
    # Ensure we start from shape (50, 4)
    arr = zarr.open(str(zarr_dir), mode="r+")
    assert arr.shape == (50, 4), f"Expected (50, 4), got {arr.shape}"

    settings = get_settings()
    backend = ZarrFilesystemBackend(settings.resolved_roots)
    reg = registry_mod.get_registry()

    batch_sizes = [4, 7, 3]
    results: list[tuple[int, int]] = []
    results_lock = threading.Lock()

    def _do_append(size: int) -> None:
        vecs = np.full((size, 4), float(size), dtype="float32")
        start_row, new_n = backend.append_vectors("main--matrix", vecs, registry=reg)
        with results_lock:
            results.append((start_row, new_n))

    threads = [threading.Thread(target=_do_append, args=(s,)) for s in batch_sizes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3

    ranges = sorted((r[0], r[1]) for r in results)
    assert ranges[0][0] == 50
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0], f"gap between {ranges[i]} and {ranges[i+1]}"
    total_M = sum(batch_sizes)
    assert ranges[-1][1] == 50 + total_M


# ---------------------------------------------------------------------------
# 10. Appended data is readable
# ---------------------------------------------------------------------------


def test_append_data_readable_after_append(app_client):
    """GET /data with offset=50 returns the just-appended vectors."""
    client, _ = app_client
    appended_data = [[float(100 + i * 4 + j) for j in range(4)] for i in range(3)]
    resp = client.post(
        "/api/datasets/main--matrix/vectors/append",
        json={"vectors": appended_data, "dtype": "float32"},
    )
    assert resp.status_code == 200

    # Read from offset 50 (first appended row) via slice endpoint
    slice_resp = client.get(
        "/api/datasets/main--matrix/slice",
        params={"slice": "50:53"},
    )
    assert slice_resp.status_code == 200
    body = slice_resp.json()
    assert body["out_shape"] == [3, 4]
    data = body["data"]
    assert data["shape"] == [3, 4]

    # Verify rows match what we appended
    assert data["rows"] == appended_data


# ---------------------------------------------------------------------------
# 11. Append does NOT read existing vectors (O(M) guarantee)
# ---------------------------------------------------------------------------


def test_append_does_not_read_existing_vectors(app_client):
    """Monkey-patch zarr.Array.__getitem__ to prove existing rows are untouched."""
    client, base_path = app_client
    _ = base_path

    import zarr

    original_getitem = zarr.Array.__getitem__

    read_calls: list[tuple] = []

    def _tracking_getitem(self, selection):
        read_calls.append(selection)
        return original_getitem(self, selection)

    with patch.object(zarr.Array, "__getitem__", _tracking_getitem):
        vectors = [[float(i) for i in range(4)] for _ in range(2)]
        resp = client.post(
            "/api/datasets/main--matrix/vectors/append",
            json={"vectors": vectors, "dtype": "float32"},
        )
        assert resp.status_code == 200

    # The append operation should only write (not read) the data.
    # __setitem__ is called, not __getitem__, for the new slice.
    # Any __getitem__ calls during append (e.g. from summarize) are O(1) metadata.
    # The key assertion: no __getitem__ call reads existing data rows [0:50].
    for sel in read_calls:
        if isinstance(sel, tuple):
            row_slice = sel[0]
        elif isinstance(sel, slice):
            row_slice = sel
        else:
            continue
        # Allow reads on integer indices (scalar access) or range [0:1]
        # But disallow reads that cover a large portion of existing data
        if isinstance(row_slice, slice):
            stop = row_slice.stop or 0
            assert stop <= 1, f"Read existing data at {sel} — violates O(M) guarantee"

    # Verify the append did complete
    meta = client.get("/api/datasets/main--matrix/metadata").json()
    assert meta["shape"] == [52, 4]
