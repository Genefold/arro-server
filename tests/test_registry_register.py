"""tests/test_registry_register.py

Tests for StorageRegistry.register_dataset() and related cache behaviour.

Covers:
    1. register_dataset() makes a new dataset visible without rescan
    2. register_dataset() does not evict pre-existing cached entries
    3. register_dataset() on a cold cache triggers lazy-load first
    4. invalidate() forces rescan on next list_datasets()
    5. reset_registry_cache() delegates to invalidate() (not cache_clear)
    6. Concurrent register_dataset() calls do not corrupt _cache
    7. StorageBackend Protocol compliance: ZarrFilesystemBackend.summarize()
    8. summarize() raises DatasetNotFound for non-existent path
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from arro_server.storage.base import DatasetSummary, StorageBackend
from arro_server.storage.registry import StorageRegistry, get_registry, reset_registry_cache
from arro_server.storage.zarr_fs import ZarrFilesystemBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_mock_backend(summaries: list[DatasetSummary]) -> MagicMock:
    backend = MagicMock(spec=StorageBackend)
    backend.name = "mock"
    backend.list_datasets.return_value = summaries
    backend._roots = {"main": Path("/fake/main")}
    return backend


def _write_zarr_array(path: Path) -> None:
    """Write a minimal valid Zarr v3 array to path."""
    arr = np.arange(40, dtype=np.float64).reshape(10, 4)
    zarr.save(str(path), arr)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure get_registry singleton is fully reset between tests."""
    get_registry.cache_clear()
    yield
    get_registry.cache_clear()


# ---------------------------------------------------------------------------
# 1. register_dataset makes new dataset visible without rescan
# ---------------------------------------------------------------------------


def test_register_dataset_visible_in_list(tmp_path: Path) -> None:
    """After register_dataset(), the new dataset appears in list_datasets()."""
    existing = _make_summary("main--existing")
    backend = _make_mock_backend([existing])

    # Make summarize() return a predictable summary for the new dataset
    new_summary = _make_summary("main--new")
    backend.summarize.return_value = new_summary

    registry = StorageRegistry([backend])

    # Warm up the cache
    datasets = registry.list_datasets()
    assert len(datasets) == 1
    assert backend.list_datasets.call_count == 1  # one scan on cold cache

    # Register new dataset — must NOT trigger another scan
    registry.register_dataset("main--new", tmp_path / "new.zarr")
    assert backend.list_datasets.call_count == 1  # no extra scan

    datasets = registry.list_datasets()
    ids = [d.dataset_id for d in datasets]
    assert "main--new" in ids
    assert "main--existing" in ids


# ---------------------------------------------------------------------------
# 2. register_dataset does not evict pre-existing entries
# ---------------------------------------------------------------------------


def test_register_dataset_does_not_evict(tmp_path: Path) -> None:
    """register_dataset() inserts without evicting other cached entries."""
    summaries = [_make_summary(f"main--ds-{i}") for i in range(5)]
    backend = _make_mock_backend(summaries)
    backend.summarize.return_value = _make_summary("main--new")

    registry = StorageRegistry([backend])
    registry.list_datasets()  # warm cache

    registry.register_dataset("main--new", tmp_path / "new.zarr")

    result_ids = {d.dataset_id for d in registry.list_datasets()}
    for i in range(5):
        assert f"main--ds-{i}" in result_ids
    assert "main--new" in result_ids


# ---------------------------------------------------------------------------
# 3. register_dataset on cold cache triggers lazy-load
# ---------------------------------------------------------------------------


def test_register_dataset_cold_cache_lazy_loads(tmp_path: Path) -> None:
    """If called before list_datasets(), register_dataset triggers a full scan first.

    This prevents silent omission of pre-existing datasets.
    """
    existing = _make_summary("main--pre-existing")
    backend = _make_mock_backend([existing])
    backend.summarize.return_value = _make_summary("main--new")

    registry = StorageRegistry([backend])
    # Do NOT call list_datasets() first — cache is None

    registry.register_dataset("main--new", tmp_path / "new.zarr")

    # list_datasets() must not trigger another scan (cache already populated)
    call_count_before = backend.list_datasets.call_count
    datasets = registry.list_datasets()
    assert backend.list_datasets.call_count == call_count_before  # no extra scan

    ids = {d.dataset_id for d in datasets}
    assert "main--pre-existing" in ids  # lazy-load captured pre-existing
    assert "main--new" in ids


# ---------------------------------------------------------------------------
# 4. invalidate() forces rescan on next list_datasets()
# ---------------------------------------------------------------------------


def test_invalidate_forces_rescan(tmp_path: Path) -> None:
    """invalidate() sets _cache = None; next list_datasets() rescans backends."""
    backend = _make_mock_backend([_make_summary("main--a")])
    registry = StorageRegistry([backend])

    registry.list_datasets()  # first scan
    assert backend.list_datasets.call_count == 1

    registry.invalidate()

    registry.list_datasets()  # must rescan
    assert backend.list_datasets.call_count == 2


def test_invalidate_does_not_destroy_singleton() -> None:
    """invalidate() must not destroy the get_registry() singleton."""
    registry_before = get_registry()
    reset_registry_cache()  # calls get_registry().invalidate()
    registry_after = get_registry()
    assert registry_before is registry_after  # same object


# ---------------------------------------------------------------------------
# 5. reset_registry_cache delegates to invalidate, not cache_clear
# ---------------------------------------------------------------------------


def test_reset_registry_cache_is_invalidate_not_cache_clear() -> None:
    """reset_registry_cache() must preserve the singleton."""
    r1 = get_registry()
    reset_registry_cache()
    r2 = get_registry()
    assert r1 is r2, "reset_registry_cache() must not destroy the singleton"


# ---------------------------------------------------------------------------
# 6. Concurrent register_dataset() calls do not corrupt _cache
# ---------------------------------------------------------------------------


def test_concurrent_register_dataset_no_corruption(tmp_path: Path) -> None:
    """20 threads each register a unique dataset concurrently.

    All 20 must appear in list_datasets() after all threads complete.
    This is the regression test for the threading.RLock on _cache.
    """
    n = 20
    dataset_ids = [f"main--concurrent-{i}" for i in range(n)]

    backend = _make_mock_backend([])
    backend.summarize.side_effect = lambda ds_id, path: _make_summary(ds_id)

    registry = StorageRegistry([backend])
    registry.list_datasets()  # warm empty cache

    errors: list[Exception] = []
    barrier = threading.Barrier(n)

    def worker(dataset_id: str) -> None:
        barrier.wait()  # maximise contention
        try:
            registry.register_dataset(dataset_id, tmp_path / f"{dataset_id}.zarr")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(ds_id,)) for ds_id in dataset_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions during concurrent register: {errors}"

    result_ids = {d.dataset_id for d in registry.list_datasets()}
    missing = [ds_id for ds_id in dataset_ids if ds_id not in result_ids]
    assert not missing, f"Missing dataset IDs: {missing}"


# ---------------------------------------------------------------------------
# 7. Protocol compliance: ZarrFilesystemBackend implements summarize()
# ---------------------------------------------------------------------------


def test_zarr_backend_implements_storage_backend_protocol() -> None:
    """ZarrFilesystemBackend must satisfy the StorageBackend Protocol."""
    backend = ZarrFilesystemBackend({})
    assert isinstance(backend, StorageBackend), (
        "ZarrFilesystemBackend does not satisfy StorageBackend Protocol. "
        "Ensure summarize() is implemented."
    )


def test_zarr_backend_summarize_returns_correct_summary(tmp_path: Path) -> None:
    """summarize() on a valid Zarr array returns the expected DatasetSummary."""
    zarr_path = tmp_path / "test_array.zarr"
    _write_zarr_array(zarr_path)

    backend = ZarrFilesystemBackend({"main": tmp_path})
    summary = backend.summarize("main--test_array", zarr_path)

    assert summary.dataset_id == "main--test_array"
    assert summary.root == "main"
    assert summary.shape == (10, 4)
    assert summary.dtype == "float64"
    assert summary.kind == "array"


# ---------------------------------------------------------------------------
# 8. summarize() raises DatasetNotFound for non-existent path
# ---------------------------------------------------------------------------


def test_zarr_backend_summarize_raises_for_missing_path(tmp_path: Path) -> None:
    """summarize() must raise DatasetNotFound if fs_path does not exist."""
    from arro_server.errors import DatasetNotFound

    backend = ZarrFilesystemBackend({"main": tmp_path})
    with pytest.raises(DatasetNotFound):
        backend.summarize("main--nonexistent", tmp_path / "nonexistent.zarr")
