"""Tests for DELETE /api/datasets/{dataset_id:path} (issue #22).

Design decisions tested here:
- invalidate_dataset() is O(1) and does not destroy registry singleton
- invalidate_dataset() is safe when cache is None (dirty)
- Path traversal guard raises 403 (not 400)
- rmtree FileNotFoundError treated as success (idempotency)
- Route ordering: DELETE /{id}/index is matched before DELETE /{id}
- index_deleted flag reflects actual adapter.delete_index() return value

Known limitation NOT tested (by design):
- delete-while-reading race condition: requires threading in tests and
  would be flaky. Documented in routes.py delete_dataset docstring and
  in registry.py invalidate_dataset() docstring.
  TODO(issue-22-rwlock): add stress test when RWLock is implemented.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient

from arro_server.api.routes import _assert_dataset_path_within_roots
from arro_server.storage.base import DatasetSummary
from arro_server.storage.registry import StorageRegistry, get_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_zarr_array(path: Path, shape: tuple = (10, 4), dtype: str = "float64") -> None:
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    zarr.save(str(path), arr)


def _make_summary(dataset_id: str, root: str = "main") -> DatasetSummary:
    return DatasetSummary(
        dataset_id=dataset_id,
        root=root,
        path=dataset_id.split("--", 1)[-1] if "--" in dataset_id else ".",
        shape=(10, 4),
        dtype="float64",
        chunks=(10, 4),
        kind="array",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    get_registry.cache_clear()
    from arro_server import arrowspace_adapter
    arrowspace_adapter.reset_adapter_cache()
    yield
    get_registry.cache_clear()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture()
def app_client(tmp_path):
    """TestClient with a single configured data root pointing to tmp_path/datasets."""
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.arrowspace_adapter import reset_adapter_cache

    root_dir = tmp_path / "datasets"
    root_dir.mkdir()

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={root_dir}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    get_registry.cache_clear()
    reset_adapter_cache()

    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, root_dir

    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    get_registry.cache_clear()
    reset_adapter_cache()


@pytest.fixture()
def registered_dataset(app_client):
    """Register a real zarr array via /upload/init + /upload/commit.
    Returns (client, root_dir, dataset_id).
    """
    client, root_dir = app_client
    dataset_id = "main--test-delete"
    shape = (10, 4)

    # Init
    r = client.post("/api/upload/init", json={"dataset_id": dataset_id, "root": "main"})
    assert r.status_code == 200
    upload_path = r.json()["upload_path"]

    # Write zarr
    _write_zarr_array(Path(upload_path), shape=shape)

    # Commit
    r = client.post("/api/upload/commit", json={"dataset_id": dataset_id, "fs_path": upload_path})
    assert r.status_code == 200

    return client, root_dir, dataset_id


# ---------------------------------------------------------------------------
# Group A — Happy path
# ---------------------------------------------------------------------------


def test_delete_removes_zarr_directory(registered_dataset):
    """Upload → DELETE → the zarr directory no longer exists on disk."""
    client, root_dir, dataset_id = registered_dataset
    zarr_path = root_dir / "test-delete"
    assert zarr_path.exists()

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200

    assert not zarr_path.exists()


def test_delete_returns_correct_response(registered_dataset):
    """Response body is {"id": ..., "deleted": True, "index_deleted": False}."""
    client, _root_dir, dataset_id = registered_dataset

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200
    body = r.json()
    assert body == {"id": dataset_id, "deleted": True, "index_deleted": False}


def test_delete_evicts_from_registry_cache(registered_dataset):
    """After DELETE, list_datasets() does not contain the dataset_id."""
    client, _root_dir, dataset_id = registered_dataset

    # Confirm it appears in listing before delete
    r = client.get("/api/datasets")
    ids_before = {d["id"] for d in r.json()["datasets"]}
    assert dataset_id in ids_before

    # Delete
    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200

    # Must not appear in listing
    r = client.get("/api/datasets")
    ids_after = {d["id"] for d in r.json()["datasets"]}
    assert dataset_id not in ids_after


def test_delete_then_get_datasets_returns_404(registered_dataset):
    """GET /datasets/{id}/metadata after DELETE returns 404."""
    client, _root_dir, dataset_id = registered_dataset

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200

    r = client.get(f"/api/datasets/{dataset_id}/metadata")
    assert r.status_code == 404


def test_delete_with_index_sets_index_deleted_true(registered_dataset, monkeypatch):
    """If an index exists (adapter.delete_index returns True), index_deleted is True."""
    client, _root_dir, dataset_id = registered_dataset

    # Patch delete_index on the concrete adapter class used at runtime.
    # _ArrowSpaceAdapter is used when arrowspace is installed; _SidecarAdapter
    # is the fallback. We patch both to ensure the test works regardless.
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter, _SidecarAdapter

    def mock_delete_index(self, dataset_id, index_store):
        return True

    monkeypatch.setattr(_ArrowSpaceAdapter, "delete_index", mock_delete_index)
    monkeypatch.setattr(_SidecarAdapter, "delete_index", mock_delete_index)

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["index_deleted"] is True


# ---------------------------------------------------------------------------
# Group B — Error cases
# ---------------------------------------------------------------------------


def test_delete_missing_dataset_returns_404(app_client):
    """DELETE on a dataset that was never registered → HTTP 404."""
    client, _root_dir = app_client

    r = client.delete("/api/datasets/main--nonexistent")
    assert r.status_code == 404
    body = r.json()
    assert "not found" in body["detail"].lower()


def test_delete_twice_returns_404_on_second(registered_dataset):
    """DELETE twice on same id → second call returns 404."""
    client, _root_dir, dataset_id = registered_dataset

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 404


def test_delete_path_traversal_returns_403(registered_dataset, monkeypatch):
    """Dataset whose resolved fs_path is outside all data roots → HTTP 403.

    Mocks h.fs_path directly so the test does not depend on whether
    ZarrFilesystemBackend resolves symlinks during open(). The unit
    under test is _assert_dataset_path_within_roots, not the backend.
    """
    from arro_server.storage import zarr_fs

    client, root_dir, dataset_id = registered_dataset

    outside_path = root_dir.parent / "outside_zarr" / "test-delete"
    outside_path.parent.mkdir(exist_ok=True)

    original_open = zarr_fs.ZarrFilesystemBackend.open

    def patched_open(self, did):
        handle = original_open(self, did)
        # Override fs_path to simulate a dataset that resolved outside roots
        handle.fs_path = outside_path
        return handle

    monkeypatch.setattr(zarr_fs.ZarrFilesystemBackend, "open", patched_open)

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 403
    assert "outside" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Group C — invalidate_dataset unit tests
# ---------------------------------------------------------------------------


def test_invalidate_dataset_removes_from_populated_cache():
    """After list_datasets() populates cache, invalidate_dataset(id) removes only that id."""
    s1 = _make_summary("main--a")
    s2 = _make_summary("main--b")
    s3 = _make_summary("main--c")
    backend = MagicMock()
    backend.name = "mock"
    backend.list_datasets.return_value = [s1, s2, s3]
    backend._roots = {"main": Path("/fake")}

    reg = StorageRegistry([backend])
    reg.list_datasets()  # populate cache

    reg.invalidate_dataset("main--b")

    ids = {d.dataset_id for d in reg.list_datasets()}
    assert "main--b" not in ids
    assert "main--a" in ids
    assert "main--c" in ids
    # No extra rescan was triggered
    assert backend.list_datasets.call_count == 1


def test_invalidate_dataset_on_dirty_cache_is_noop():
    """If cache is None (dirty), invalidate_dataset() does not raise and cache stays None."""
    s1 = _make_summary("main--a")
    backend = MagicMock()
    backend.name = "mock"
    backend.list_datasets.return_value = [s1]
    backend._roots = {"main": Path("/fake")}

    reg = StorageRegistry([backend])
    # cache is None — never called list_datasets()

    # Must not raise
    reg.invalidate_dataset("main--a")

    # list_datasets() must still trigger a full rescan (cache was None)
    reg.list_datasets()
    assert backend.list_datasets.call_count == 1


def test_invalidate_dataset_unknown_id_is_noop():
    """invalidate_dataset('nonexistent') on populated cache does not raise."""
    s1 = _make_summary("main--a")
    backend = MagicMock()
    backend.name = "mock"
    backend.list_datasets.return_value = [s1]
    backend._roots = {"main": Path("/fake")}

    reg = StorageRegistry([backend])
    reg.list_datasets()

    reg.invalidate_dataset("nonexistent")
    # no exception


def test_invalidate_dataset_does_not_destroy_singleton():
    """The registry singleton remains the same object after invalidate_dataset."""
    reg_before = get_registry()
    reg_before.invalidate_dataset("anything")
    reg_after = get_registry()
    assert reg_before is reg_after


# ---------------------------------------------------------------------------
# Group D — Idempotency
# ---------------------------------------------------------------------------


def test_delete_already_gone_from_disk_succeeds(registered_dataset, monkeypatch):
    """shutil.rmtree FileNotFoundError is caught → DELETE succeeds.

    This tests the idempotency guarantee in the rmtree step: if the zarr
    directory disappears (e.g. concurrent deletion) between step 3
    (invalidate_dataset) and step 5 (shutil.rmtree), the FileNotFoundError
    is caught and treated as success.
    """
    client, root_dir, dataset_id = registered_dataset
    zarr_path = root_dir / "test-delete"
    assert zarr_path.exists()

    import shutil as shutil_mod

    def raisy_rmtree(path):
        raise FileNotFoundError(f"No such file: {path}")

    monkeypatch.setattr(shutil_mod, "rmtree", raisy_rmtree)

    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200
    body = r.json()
    assert body == {"id": dataset_id, "deleted": True, "index_deleted": False}


# ---------------------------------------------------------------------------
# Group E — Route ordering regression
# ---------------------------------------------------------------------------


def test_delete_index_route_still_works_after_delete_dataset_added(registered_dataset):
    """DELETE /datasets/{id}/index still works and does NOT delete the dataset.

    Regression test: ensure the route ordering (delete_index before
    delete_dataset) is maintained so that /index sub-routes are matched
    before the catch-all /{id}.

    We check that the request reaches the delete_index handler by verifying
    the response is NOT 200 (which would mean delete_dataset handled it).
    The actual status code depends on the adapter: 404 when _ArrowSpaceAdapter
    is active (no index found), 503 when _SidecarAdapter is active
    (OptionalDependencyMissing). Either is acceptable — both prove the
    correct route was matched.
    """
    client, _root_dir, dataset_id = registered_dataset

    # DELETE /{id}/index must NOT return 200 (if it did, the dataset would
    # have been deleted by the catch-all delete_dataset handler instead).
    r = client.delete(f"/api/datasets/{dataset_id}/index")
    assert r.status_code != 200, \
        "DELETE /{id}/index matched delete_dataset handler — route ordering broken"

    # Verify the dataset is still there (delete_index did NOT cascade to the dataset)
    r = client.get(f"/api/datasets/{dataset_id}/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == dataset_id

    # Now delete the dataset properly and verify
    r = client.delete(f"/api/datasets/{dataset_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get(f"/api/datasets/{dataset_id}/metadata")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# _assert_dataset_path_within_roots unit tests
# ---------------------------------------------------------------------------


def test_assert_dataset_path_within_roots_allows_inside():
    """Path inside a root is allowed (no exception)."""
    roots = {"main": Path("/tmp/main")}
    _assert_dataset_path_within_roots(Path("/tmp/main/dataset.zarr"), roots)


def test_assert_dataset_path_within_roots_raises_403_for_outside():
    """Path outside all roots raises HTTP 403."""
    roots = {"main": Path("/tmp/main")}
    with pytest.raises(Exception) as exc_info:
        _assert_dataset_path_within_roots(Path("/tmp/other/dataset.zarr"), roots)
    assert exc_info.value.status_code == 403  # type: ignore[union-attr]
