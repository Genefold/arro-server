"""tests/test_thread_safety.py — Concurrency safety for _LRUIndexCache and manifest writes.

Covers:
  * _LRUIndexCache under concurrent get/put/delete from multiple threads
  * Concurrent build_index calls do not drop manifest entries
  * Concurrent delete_index calls do not corrupt manifest
  * _LRUIndexCache.keys() returns a stable snapshot (no mutation-during-iteration)
"""

from __future__ import annotations

import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

NITEMS = 10
NFEATURES = 4
NCLUSTERS = 2
GRAPH_PARAMS = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
FIXTURE_ARRAY = np.arange(NITEMS * NFEATURES, dtype=np.float64).reshape(NITEMS, NFEATURES)

# ---------------------------------------------------------------------------
# Helpers — mirrors test_phase2_persistence.py
# ---------------------------------------------------------------------------


def _make_fake_aspace() -> MagicMock:
    aspace = MagicMock()
    aspace.nitems = NITEMS
    aspace.nfeatures = NFEATURES
    aspace.nclusters = NCLUSTERS
    return aspace


def _make_fake_gl() -> MagicMock:
    gl = MagicMock()
    gl.nnodes = NITEMS
    gl.shape = (NITEMS, NITEMS)
    gl.graph_params = GRAPH_PARAMS
    n = NITEMS
    gl.to_csr.return_value = (
        np.ones(n, dtype=np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64),
        (n, n),
    )
    return gl


def _make_fake_mod() -> types.ModuleType:
    fake_mod = types.ModuleType("arrowspace")
    aspace = _make_fake_aspace()
    gl = _make_fake_gl()

    class FakeBuilder:
        def build(self, graph_params, array):
            return aspace, gl

    fake_mod.ArrowSpaceBuilder = FakeBuilder  # type: ignore[attr-defined]
    return fake_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mod() -> types.ModuleType:
    return _make_fake_mod()


@pytest.fixture
def adapter(fake_mod):
    from arro_server.arrowspace_adapter import _ArrowSpaceAdapter

    return _ArrowSpaceAdapter(fake_mod, cache_size=32)


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    d = tmp_path / "index_store"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# 1. _LRUIndexCache thread safety (unit)
# ---------------------------------------------------------------------------


class TestLRUCacheThreadSafety:
    """All tests use threading.Thread directly — no pytest-xdist or async."""

    def _make_entry(self):
        from arro_server.arrowspace_adapter import _IndexEntry

        return _IndexEntry(aspace=None, gl=None, nitems=1, nfeatures=1, nclusters=1)

    def test_concurrent_puts_no_corruption(self):
        """100 threads each put a unique key — all keys must be present."""
        from arro_server.arrowspace_adapter import _LRUIndexCache

        n_threads = 100
        cache = _LRUIndexCache(maxsize=n_threads + 10)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                cache.put(f"key-{i}", self._make_entry())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent puts: {errors}"
        assert len(cache.keys()) == n_threads

    def test_concurrent_gets_no_exception(self):
        """50 threads reading the same key concurrently must not raise."""
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=4)
        cache.put("shared", self._make_entry())
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    cache.get("shared")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_put_and_delete_no_exception(self):
        """Interleaved puts and deletes on the same key must not raise."""
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=10)
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def putter() -> None:
            barrier.wait()
            try:
                for _ in range(200):
                    cache.put("contested", self._make_entry())
            except Exception as exc:
                errors.append(exc)

        def deleter() -> None:
            barrier.wait()
            try:
                for _ in range(200):
                    cache.delete("contested")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=putter)
        t2 = threading.Thread(target=deleter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

    def test_keys_snapshot_is_stable(self):
        """keys() must return a copy — mutations after the call must not affect it."""
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=10)
        cache.put("a", self._make_entry())
        cache.put("b", self._make_entry())

        snapshot = cache.keys()
        cache.put("c", self._make_entry())  # mutate after snapshot

        assert "c" not in snapshot  # snapshot is a copy, not a live view

    def test_eviction_under_concurrent_load_no_exception(self):
        """LRU eviction triggered from multiple threads must not corrupt internal state."""
        from arro_server.arrowspace_adapter import _LRUIndexCache

        cache = _LRUIndexCache(maxsize=5)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                cache.put(f"key-{i}", self._make_entry())
                cache.get(f"key-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache.keys()) <= 5  # eviction kept cache within bounds


# ---------------------------------------------------------------------------
# 2. Manifest write safety (unit)
# ---------------------------------------------------------------------------


class TestManifestConcurrency:
    """Concurrent build_index and delete_index must not lose or corrupt manifest entries."""

    def test_concurrent_builds_no_lost_entries(self, adapter, tmp_store: Path):
        """10 concurrent build_index calls each on a different dataset_id.

        Every dataset_id must appear in the manifest after all threads complete.
        This is the core regression test for the _MANIFEST_LOCK fix.
        """
        from arro_server.arrowspace_adapter import _read_manifest

        n = 10
        dataset_ids = [f"root--dataset-{i}" for i in range(n)]
        errors: list[Exception] = []
        barrier = threading.Barrier(n)

        def worker(dataset_id: str) -> None:
            barrier.wait()  # synchronise all threads to maximise contention
            try:
                adapter.build_index(dataset_id, FIXTURE_ARRAY.copy(), tmp_store)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(ds_id,)) for ds_id in dataset_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent builds: {errors}"

        manifest = _read_manifest(tmp_store)
        missing = [ds_id for ds_id in dataset_ids if ds_id not in manifest]
        assert not missing, f"Lost manifest entries: {missing}"

    def test_concurrent_builds_unique_dataset_names(self, adapter, tmp_store: Path):
        """Each concurrent build must produce a unique dataset_name in the manifest."""
        from arro_server.arrowspace_adapter import _read_manifest

        n = 10
        dataset_ids = [f"root--ds-name-{i}" for i in range(n)]
        barrier = threading.Barrier(n)

        def worker(dataset_id: str) -> None:
            barrier.wait()
            adapter.build_index(dataset_id, FIXTURE_ARRAY.copy(), tmp_store)

        threads = [threading.Thread(target=worker, args=(ds_id,)) for ds_id in dataset_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        manifest = _read_manifest(tmp_store)
        names = [manifest[ds_id]["dataset_name"] for ds_id in dataset_ids if ds_id in manifest]
        assert len(names) == len(set(names)), f"Duplicate dataset_names: {names}"

    def test_concurrent_build_and_delete_no_corruption(self, adapter, tmp_store: Path):
        """Interleaved build and delete on different datasets must not corrupt manifest."""
        from arro_server.arrowspace_adapter import _read_manifest

        build_ids = [f"root--build-{i}" for i in range(5)]
        delete_ids = [f"root--delete-{i}" for i in range(5)]

        # Pre-build the datasets that will be deleted
        for ds_id in delete_ids:
            adapter.build_index(ds_id, FIXTURE_ARRAY.copy(), tmp_store)

        errors: list[Exception] = []
        barrier = threading.Barrier(len(build_ids) + len(delete_ids))

        def builder(dataset_id: str) -> None:
            barrier.wait()
            try:
                adapter.build_index(dataset_id, FIXTURE_ARRAY.copy(), tmp_store)
            except Exception as exc:
                errors.append(exc)

        def deleter(dataset_id: str) -> None:
            barrier.wait()
            try:
                adapter.delete_index(dataset_id, tmp_store)
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=builder, args=(ds_id,)) for ds_id in build_ids]
            + [threading.Thread(target=deleter, args=(ds_id,)) for ds_id in delete_ids]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        manifest = _read_manifest(tmp_store)
        # Newly built entries must be present
        missing_builds = [ds_id for ds_id in build_ids if ds_id not in manifest]
        assert not missing_builds, f"Lost build entries: {missing_builds}"
        # Deleted entries must be absent
        surviving_deletes = [ds_id for ds_id in delete_ids if ds_id in manifest]
        assert not surviving_deletes, f"Delete entries survived: {surviving_deletes}"

    def test_sequential_builds_still_correct_after_lock_added(self, adapter, tmp_store: Path):
        """Regression: sequential build behaviour must be identical to pre-lock baseline.

        This test mirrors TestBuildIndexManifest.test_multiple_datasets_in_manifest
        from test_phase2_persistence.py to confirm the lock introduces no regression
        in the single-threaded path.
        """
        from arro_server.arrowspace_adapter import _read_manifest

        adapter.build_index("ds_seq_a", FIXTURE_ARRAY.copy(), tmp_store)
        adapter.build_index("ds_seq_b", FIXTURE_ARRAY.copy(), tmp_store)

        manifest = _read_manifest(tmp_store)
        assert "ds_seq_a" in manifest
        assert "ds_seq_b" in manifest
        assert manifest["ds_seq_a"]["dataset_name"] != manifest["ds_seq_b"]["dataset_name"]
