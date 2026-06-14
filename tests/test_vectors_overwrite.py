"""Tests for POST /api/datasets/{dataset_id}/vectors/overwrite.

Test index:
    1.  test_overwrite_single_row              — single update, value readable via /slice
    2.  test_overwrite_multiple_rows           — K=3 updates, all values correct
    3.  test_overwrite_shape_unchanged         — shape in /metadata is identical before/after
    4.  test_overwrite_row_index_out_of_bounds_422  — row_index >= N → 422, no partial write
    5.  test_overwrite_row_index_negative_422  — row_index < 0 → 422 from Pydantic
    6.  test_overwrite_dim_mismatch_422        — wrong vector length → 422
    7.  test_overwrite_empty_updates_422       — empty updates list → 422 from Pydantic
    8.  test_overwrite_dataset_not_found_404   — non-existent dataset_id → 404
    9.  test_overwrite_float64_to_float32_auto_cast — same_kind cast accepted
    10. test_overwrite_incompatible_dtype_422  — complex64 → float32 rejected → 422
    11. test_overwrite_validate_all_then_write — invalid entry after valid one → 422, no partial write
    12. test_overwrite_duplicate_row_index     — last entry wins, no error
    13. test_overwrite_toctou_shape_change_422 — mock zarr.open("r+") to trigger TOCTOU
    14. test_overwrite_index_stale_warning     — (docstring only, not runtime assertion)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client(tmp_path: Path):
    """TestClient with a single float32 2-D Zarr array at main--matrix.

    Shape (50, 4), dtype float32, values arange(200).reshape(50, 4).
    """
    import zarr

    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache
    from arro_server.storage import registry as registry_mod

    zarr_dir = tmp_path / "matrix"
    arr = zarr.open(str(zarr_dir), mode="w", shape=(50, 4), chunks=(10, 4), dtype="float32")
    arr[:] = np.arange(200, dtype="float32").reshape(50, 4)

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_path}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, tmp_path

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1. Single row overwrite
# ---------------------------------------------------------------------------


def test_overwrite_single_row(app_client):
    """Overwrite row 10 and verify via GET /slice."""
    client, _ = app_client
    new_vec = [99.0, 98.0, 97.0, 96.0]
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 10, "vector": new_vec}]},
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json() == {"overwritten": 1}

    slice_resp = client.get("/api/datasets/main--matrix/slice", params={"slice": "10:11"})
    assert slice_resp.status_code == 200
    rows = slice_resp.json()["data"]["rows"]
    assert len(rows) == 1
    for got, exp in zip(rows[0], new_vec, strict=True):
        assert abs(got - exp) < 1e-6


# ---------------------------------------------------------------------------
# 2. Multiple rows
# ---------------------------------------------------------------------------


def test_overwrite_multiple_rows(app_client):
    """Overwrite rows 0, 25, 49 in a single batch."""
    client, _ = app_client
    updates = [
        {"row_index": 0,  "vector": [1.0, 2.0, 3.0, 4.0]},
        {"row_index": 25, "vector": [5.0, 6.0, 7.0, 8.0]},
        {"row_index": 49, "vector": [9.0, 10.0, 11.0, 12.0]},
    ]
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": updates},
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["overwritten"] == 3

    for upd in updates:
        r = upd["row_index"]
        slice_resp = client.get(
            "/api/datasets/main--matrix/slice", params={"slice": f"{r}:{r + 1}"}
        )
        rows = slice_resp.json()["data"]["rows"]
        for got, exp in zip(rows[0], upd["vector"], strict=True):
            assert abs(got - exp) < 1e-6


# ---------------------------------------------------------------------------
# 3. Shape unchanged
# ---------------------------------------------------------------------------


def test_overwrite_shape_unchanged(app_client):
    """Shape reported by /metadata is identical before and after overwrite."""
    client, _ = app_client
    meta_before = client.get("/api/datasets/main--matrix/metadata").json()
    client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 5, "vector": [0.0, 0.0, 0.0, 0.0]}]},
    )
    meta_after = client.get("/api/datasets/main--matrix/metadata").json()
    assert meta_before["shape"] == meta_after["shape"]


# ---------------------------------------------------------------------------
# 4. Out-of-bounds row_index → 422
# ---------------------------------------------------------------------------


def test_overwrite_row_index_out_of_bounds_422(app_client):
    """row_index == N (= 50) is out of bounds → 422."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 50, "vector": [1.0, 2.0, 3.0, 4.0]}]},
    )
    assert resp.status_code == 422
    assert "out of bounds" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 5. Negative row_index → 422 (Pydantic ge=0)
# ---------------------------------------------------------------------------


def test_overwrite_row_index_negative_422(app_client):
    """row_index < 0 is rejected by Pydantic before reaching the backend."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": -1, "vector": [1.0, 2.0, 3.0, 4.0]}]},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. Dimension mismatch → 422
# ---------------------------------------------------------------------------


def test_overwrite_dim_mismatch_422(app_client):
    """Vector with wrong dimension → 422."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 0, "vector": [1.0, 2.0, 3.0]}]},  # D=3, expects 4
    )
    assert resp.status_code == 422
    assert "dimension mismatch" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. Empty updates → 422 (Pydantic min_length=1)
# ---------------------------------------------------------------------------


def test_overwrite_empty_updates_422(app_client):
    """Empty updates list is rejected by Pydantic."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": []},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 8. Dataset not found → 404
# ---------------------------------------------------------------------------


def test_overwrite_dataset_not_found_404(app_client):
    """Non-existent dataset_id → 404."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--nonexistent/vectors/overwrite",
        json={"updates": [{"row_index": 0, "vector": [1.0, 2.0, 3.0, 4.0]}]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. Auto-cast float64 → float32 (same_kind)
# ---------------------------------------------------------------------------


def test_overwrite_float64_to_float32_auto_cast(app_client):
    """float64 vectors (default dtype) are silently cast to float32 dataset."""
    client, _ = app_client
    # No explicit dtype — route defaults to float64; dataset is float32 → same_kind cast
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 3, "vector": [10.0, 20.0, 30.0, 40.0]}]},
    )
    assert resp.status_code == 200, resp.json()
    slice_resp = client.get("/api/datasets/main--matrix/slice", params={"slice": "3:4"})
    rows = slice_resp.json()["data"]["rows"]
    for got, exp in zip(rows[0], [10.0, 20.0, 30.0, 40.0], strict=True):
        assert abs(got - exp) < 1e-6


# ---------------------------------------------------------------------------
# 10. Incompatible dtype → 422
# ---------------------------------------------------------------------------


def test_overwrite_incompatible_dtype_422(app_client):
    """complex64 → float32 is not same_kind → 422."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={"updates": [{"row_index": 0, "vector": [1.0, 2.0, 3.0, 4.0]}], "dtype": "complex64"},
    )
    assert resp.status_code == 422
    assert "dtype mismatch" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 11. Validate-all-then-write: invalid entry after valid ones → no partial write
# ---------------------------------------------------------------------------


def test_overwrite_validate_all_then_write(app_client):
    """If updates[1] is invalid, updates[0] must NOT have been written."""
    client, _ = app_client
    # Read original value at row 0
    orig = client.get("/api/datasets/main--matrix/slice", params={"slice": "0:1"}).json()
    orig_row = orig["data"]["rows"][0]

    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={
            "updates": [
                {"row_index": 0,  "vector": [99.0, 99.0, 99.0, 99.0]},  # valid
                {"row_index": 50, "vector": [1.0, 2.0, 3.0, 4.0]},      # out of bounds
            ]
        },
    )
    assert resp.status_code == 422

    # Row 0 must be unchanged
    after = client.get("/api/datasets/main--matrix/slice", params={"slice": "0:1"}).json()
    assert after["data"]["rows"][0] == orig_row


# ---------------------------------------------------------------------------
# 12. Duplicate row_index — last entry wins
# ---------------------------------------------------------------------------


def test_overwrite_duplicate_row_index(app_client):
    """Duplicate row_index: last write wins, no error raised."""
    client, _ = app_client
    resp = client.post(
        "/api/datasets/main--matrix/vectors/overwrite",
        json={
            "updates": [
                {"row_index": 7, "vector": [1.0, 1.0, 1.0, 1.0]},
                {"row_index": 7, "vector": [2.0, 2.0, 2.0, 2.0]},  # wins
            ]
        },
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["overwritten"] == 2

    slice_resp = client.get("/api/datasets/main--matrix/slice", params={"slice": "7:8"})
    rows = slice_resp.json()["data"]["rows"]
    for val in rows[0]:
        assert abs(val - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# 13. TOCTOU — shape change between validation and write
# ---------------------------------------------------------------------------


def test_overwrite_toctou_shape_change_422(app_client):
    """If array shape changed between pre-lock validation and write, VectorShapeMismatch."""
    import zarr

    from arro_server.api.schemas import RowUpdate
    from arro_server.settings import get_settings
    from arro_server.storage.zarr_fs import ZarrFilesystemBackend

    _, _ = app_client
    settings = get_settings()
    backend = ZarrFilesystemBackend(settings.resolved_roots)

    updates = [RowUpdate(row_index=0, vector=[1.0, 2.0, 3.0, 4.0])]

    original_open = zarr.open

    def _bait_and_switch(path, mode="r", **kw):
        arr = original_open(path, mode=mode, **kw)
        if mode == "r+":
            mock_arr = MagicMock(spec=zarr.Array)
            mock_arr.ndim = 2
            mock_arr.shape = (50, 7)          # D changed from 4 to 7
            mock_arr.dtype = arr.dtype
            return mock_arr
        return arr

    with patch("zarr.open", _bait_and_switch):
        with pytest.raises(Exception) as exc_info:
            backend.overwrite_vectors("main--matrix", updates, dtype="float32")
        assert "shape changed" in str(exc_info.value).lower()
