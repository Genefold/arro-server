"""tests/test_upload.py

Integration and unit tests for POST /api/upload/init and POST /api/upload/commit.

Test inventory:
    1.  upload_init — valid request returns correct upload_path
    2.  upload_init — unknown root returns 404
    3.  upload_init — label mismatch returns 400
    4.  upload_init — dataset_id without path component returns 400
    5.  upload_init — dataset_id with invalid characters returns 422
    6.  upload_commit — valid Zarr array registers and appears in GET /datasets
    7.  upload_commit — path outside roots returns 400 (path traversal guard)
    8.  upload_commit — non-existent Zarr path returns 404
    9.  upload_commit — overwrite of existing dataset returns index_stale=True
    10. upload_commit — incomplete Zarr (empty shape) returns 422

Security:
    11. _assert_path_within_roots — path inside root is allowed
    12. _assert_path_within_roots — path outside all roots raises 400
    13. _assert_path_within_roots — symlink traversal attempt raises 400
    14. _validate_zarr_summary — valid summary does not raise
    15. _validate_zarr_summary — empty shape raises 422
    16. _validate_zarr_summary — all-zero shape raises 422
    17. _validate_zarr_summary — empty dtype raises 422
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient

from arro_server.api.routes import _assert_path_within_roots, _validate_zarr_summary
from arro_server.storage import registry as registry_mod
from arro_server.storage.base import DatasetHandle, DatasetSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_zarr_array(path: Path, shape: tuple = (100, 32), dtype: str = "float32") -> None:
    """Write a minimal valid Zarr array to path."""
    arr = np.zeros(shape, dtype=dtype)
    zarr.save(str(path), arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client(tmp_path):
    """TestClient with a configured data root pointing to tmp_path."""
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_path}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, tmp_path

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()
    reset_adapter_cache()


# ---------------------------------------------------------------------------
# 1. upload_init — valid request
# ---------------------------------------------------------------------------


def test_upload_init_valid(app_client):
    """Valid init returns 200 with correct upload_path inside the root."""
    client, root = app_client
    resp = client.post("/api/upload/init", json={"dataset_id": "main--cube", "root": "main"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "main--cube"
    assert body["root"] == "main"
    upload_path = Path(body["upload_path"])
    assert upload_path.is_relative_to(root) or str(upload_path).startswith(str(root))
    assert upload_path.name == "cube" or str(upload_path).endswith("cube")


# ---------------------------------------------------------------------------
# 2. upload_init — unknown root
# ---------------------------------------------------------------------------


def test_upload_init_unknown_root(app_client):
    """Requesting an unknown root returns 404."""
    client, _ = app_client
    resp = client.post("/api/upload/init", json={"dataset_id": "archive--cube", "root": "archive"})
    assert resp.status_code == 404
    assert "archive" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 3. upload_init — label mismatch
# ---------------------------------------------------------------------------


def test_upload_init_label_mismatch(app_client):
    """dataset_id label not matching root returns 422 (Pydantic validation)."""
    client, _ = app_client
    resp = client.post("/api/upload/init", json={"dataset_id": "other--cube", "root": "main"})
    assert resp.status_code == 422
    detail = str(resp.json()["detail"])
    assert "other" in detail


# ---------------------------------------------------------------------------
# 4. upload_init — dataset_id without path component
# ---------------------------------------------------------------------------


def test_upload_init_no_path_component(app_client):
    """dataset_id with no path component (just the label) returns 422 (Pydantic validation)."""
    client, _ = app_client
    resp = client.post("/api/upload/init", json={"dataset_id": "main", "root": "main"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "root separator" in str(detail) or "path component" in str(detail)


# ---------------------------------------------------------------------------
# 5. upload_init — invalid characters in dataset_id
# ---------------------------------------------------------------------------


def test_upload_init_invalid_dataset_id_characters(app_client):
    """dataset_id with path separators or special chars fails Pydantic validation (422)."""
    client, _ = app_client
    resp = client.post("/api/upload/init", json={"dataset_id": "main/cube", "root": "main"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. upload_commit — valid Zarr registers and appears in GET /datasets
# ---------------------------------------------------------------------------


def test_upload_commit_valid_registers_dataset(app_client):
    """Full two-phase upload: init -> write Zarr -> commit -> visible in GET /datasets."""
    client, _ = app_client

    init_resp = client.post("/api/upload/init", json={"dataset_id": "main--embeddings", "root": "main"})
    assert init_resp.status_code == 200
    upload_path = Path(init_resp.json()["upload_path"])

    _write_zarr_array(upload_path, shape=(50, 16), dtype="float64")

    commit_resp = client.post(
        "/api/upload/commit",
        json={"dataset_id": "main--embeddings", "fs_path": str(upload_path)},
    )
    assert commit_resp.status_code == 200
    body = commit_resp.json()
    assert body["registered"] is True
    assert body["dataset_id"] == "main--embeddings"
    assert body["shape"] == [50, 16]
    assert body["dtype"] == "float64"
    assert body["index_stale"] is False

    list_resp = client.get("/api/datasets")
    assert list_resp.status_code == 200
    ids = [d["id"] for d in list_resp.json()["datasets"]]
    assert "main--embeddings" in ids


# ---------------------------------------------------------------------------
# 7. upload_commit — path outside roots (path traversal guard)
# ---------------------------------------------------------------------------


def test_upload_commit_path_traversal_blocked(app_client, tmp_path):
    """fs_path outside all configured roots is rejected with 400."""
    client, _ = app_client
    outside_path = tmp_path.parent / "outside_root" / "attack.zarr"
    resp = client.post(
        "/api/upload/commit",
        json={"dataset_id": "main--attack", "fs_path": str(outside_path)},
    )
    assert resp.status_code == 400
    assert "data root" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 8. upload_commit — non-existent Zarr path
# ---------------------------------------------------------------------------


def test_upload_commit_nonexistent_path(app_client):
    """Committing a non-existent path returns 404."""
    client, root = app_client
    missing = root / "does_not_exist.zarr"
    resp = client.post(
        "/api/upload/commit",
        json={"dataset_id": "main--missing", "fs_path": str(missing)},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 9. upload_commit -- overwrite returns index_stale=True
# ---------------------------------------------------------------------------


def test_upload_commit_overwrite_returns_index_stale(app_client):
    """Committing a dataset_id that already has an index returns index_stale=True."""
    client, root = app_client
    zarr_path = root / "has_index"
    _write_zarr_array(zarr_path, shape=(10, 4))

    mock_adapter = MagicMock()
    mock_adapter.has_index.return_value = True
    with patch("arro_server.api.routes.load_arrowspace", return_value=mock_adapter):
        resp = client.post(
            "/api/upload/commit",
            json={"dataset_id": "main--has_index", "fs_path": str(zarr_path)},
        )
    assert resp.status_code == 200
    assert resp.json()["index_stale"] is True


# ---------------------------------------------------------------------------
# 10. upload_commit -- incomplete Zarr returns 422
# ---------------------------------------------------------------------------


def test_upload_commit_incomplete_zarr_returns_422(app_client):
    """Committing a dataset whose summary has empty shape returns 422."""
    client, root = app_client
    zarr_path = root / "incomplete"
    _write_zarr_array(zarr_path)

    incomplete_summary = DatasetSummary(
        dataset_id="main--incomplete",
        root="main",
        path="incomplete",
        shape=(),
        dtype="float32",
        chunks=None,
        kind="array",
    )

    incomplete_handle = DatasetHandle(
        summary=incomplete_summary,
        metadata={},
        fs_path=zarr_path,
    )

    with patch("arro_server.storage.registry.StorageRegistry.register_dataset"), \
         patch("arro_server.storage.registry.StorageRegistry.open", return_value=incomplete_handle):
        resp = client.post(
            "/api/upload/commit",
            json={"dataset_id": "main--incomplete", "fs_path": str(zarr_path)},
        )

    assert resp.status_code == 422
    assert "scalar shape" in resp.json()["detail"] or "incomplete" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 11. Security: _assert_path_within_roots -- valid path
# ---------------------------------------------------------------------------


def test__assert_path_within_roots_valid(tmp_path):
    """Path inside a root does not raise."""
    roots = {"main": tmp_path}
    nested = tmp_path / "subdir" / "array.zarr"
    _assert_path_within_roots(nested, roots)


# ---------------------------------------------------------------------------
# 12. Security: _assert_path_within_roots -- path outside all roots
# ---------------------------------------------------------------------------


def test__assert_path_within_roots_outside(tmp_path):
    """Path outside all configured roots raises HTTP 400."""
    from fastapi import HTTPException

    roots = {"main": tmp_path / "root_a"}
    outside = tmp_path / "root_b" / "array.zarr"
    with pytest.raises(HTTPException) as exc_info:
        _assert_path_within_roots(outside, roots)
    assert exc_info.value.status_code == 400
    assert "main" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 13. Security: _assert_path_within_roots -- symlink traversal attempt
# ---------------------------------------------------------------------------


def test__assert_path_within_roots_symlink_traversal(tmp_path):
    """Symlink pointing outside the root is blocked after resolve()."""
    from fastapi import HTTPException

    root_a = tmp_path / "root_a"
    root_a.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    link = root_a / "escape_link"
    link.symlink_to(outside)
    attack_path = link / "array.zarr"

    roots = {"main": root_a}
    with pytest.raises(HTTPException) as exc_info:
        _assert_path_within_roots(attack_path, roots)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# 14. Security: _validate_zarr_summary -- valid summary
# ---------------------------------------------------------------------------


def test__validate_zarr_summary_valid():
    """Valid shape and dtype do not raise."""
    _validate_zarr_summary("main--ds", (100, 32), "float32")


# ---------------------------------------------------------------------------
# 15. Security: _validate_zarr_summary -- empty shape raises 422
# ---------------------------------------------------------------------------


def test__validate_zarr_summary_empty_shape():
    """Scalar shape () raises HTTP 422."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _validate_zarr_summary("main--ds", (), "float32")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# 16. Security: _validate_zarr_summary -- all-zero shape raises 422
# ---------------------------------------------------------------------------


def test__validate_zarr_summary_zero_shape():
    """All-zero shape raises HTTP 422."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _validate_zarr_summary("main--ds", (0, 32), "float32")
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# 17. Security: _validate_zarr_summary -- empty dtype raises 422
# ---------------------------------------------------------------------------


def test__validate_zarr_summary_empty_dtype():
    """Empty dtype string raises HTTP 422."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _validate_zarr_summary("main--ds", (100, 32), "")
    assert exc_info.value.status_code == 422
