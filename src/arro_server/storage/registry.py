"""StorageRegistry — multiplexes across registered storage backends.

Caching model
-------------
The registry maintains a lazy in-process cache of DatasetSummary objects:

    _cache: dict[str, DatasetSummary] | None

* ``None``  — dirty; the next call to ``list_datasets()`` or
              ``register_dataset()`` will trigger a full O(N) backend scan
              and populate the cache.
* ``{}``    — loaded, zero datasets found.
* non-empty — loaded; dict keys are dataset_id strings.

Thread safety
-------------
All access to ``_cache`` is protected by ``_lock: threading.RLock``.
RLock (reentrant) is used for consistency with ``_LRUIndexCache`` in
``arrowspace_adapter.py`` (PR #25 fix). The reentrant property allows
``register_dataset()`` to call the lazy-load path of ``list_datasets()``
internally without deadlocking.

Invalidation
------------
``invalidate()`` sets ``_cache = None``.  The next ``list_datasets()`` call
will rescan all backends.  ``invalidate()`` does NOT destroy the singleton
returned by ``get_registry()`` — the ``@lru_cache`` on ``get_registry``
preserves the object across reloads.

``reset_registry_cache()`` is kept for backward compatibility (called by
``admin_reload`` and test teardown) and now delegates to ``invalidate()``.
Tests that need full singleton reset can still call
``get_registry.cache_clear()`` directly in teardown.

Multi-replica expansion
-----------------------
For multi-process or multi-host deployments, replace the in-process
``_lock`` with a distributed lock and publish a ``dataset_registered``
event after ``register_dataset()``:

    # TODO(multi-replica): publish event after register_dataset
    # await redis.publish("arro:dataset:registered",
    #                     json.dumps({"id": dataset_id}))
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..errors import DatasetNotFound
from ..settings import get_settings
from . import StorageBackend
from .base import DatasetHandle, DatasetSummary, decode_dataset_id
from .zarr_fs import ZarrFilesystemBackend


class StorageRegistry:
    """Multiplexes across registered storage backends with an in-process cache.

    For now we ship one backend (filesystem Zarr v3). Object-store backends
    can register themselves here without touching the API layer.
    """

    def __init__(self, backends: list[StorageBackend]) -> None:
        self._backends = backends
        # None = dirty (rescan needed). {} = loaded, zero datasets.
        self._cache: dict[str, DatasetSummary] | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[DatasetSummary]:
        """Return all known DatasetSummary objects.

        On the first call (or after invalidate()), performs a full O(N)
        scan across all backends and populates the cache.  Subsequent calls
        return the cached result in O(1).
        """
        with self._lock:
            self._ensure_loaded()
            return list(self._cache.values())  # type: ignore[union-attr]

    def open(self, dataset_id: str) -> DatasetHandle:
        """Open a dataset by ID, delegating to the first backend that knows it.

        Raises DatasetNotFound if no backend recognises dataset_id.
        """
        errors: list[str] = []
        for b in self._backends:
            try:
                return b.open(dataset_id)
            except DatasetNotFound as e:
                errors.append(str(e.detail))
                continue
        detail = " | ".join(errors) if errors else dataset_id
        raise DatasetNotFound(detail)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def register_dataset(self, dataset_id: str, fs_path: Path) -> None:
        """Insert a single dataset into the registry cache without rescanning.

        Called after POST /upload/commit to make the new dataset immediately
        visible to GET /datasets without triggering an O(N) filesystem walk.

        If the cache has not yet been populated, this method triggers a full
        scan first (lazy-load) so that the new entry is added to a complete
        cache — not an empty one — preventing silent omission of pre-existing
        datasets on the next GET /datasets call.

        The full O(N) rescan (via invalidate() + list_datasets()) is still
        recommended in a background task for eventual consistency with
        datasets written externally (e.g. arro-memory via shared volume).

        Args:
            dataset_id: URL-safe dataset ID for the new dataset.
            fs_path:    Absolute filesystem path to the Zarr node root dir.

        Raises:
            DatasetNotFound: if no backend can summarize the dataset at fs_path.

        Thread safety:
            Safe to call concurrently from multiple threads. Uses the same
            RLock as list_datasets() and invalidate().

        # TODO(multi-replica): after inserting into _cache, publish a
        # "dataset_registered" event to a message bus so other replicas can
        # update their local caches without a full scan:
        #   await redis.publish("arro:dataset:registered",
        #                       json.dumps({"id": dataset_id}))
        """
        label, _ = decode_dataset_id(dataset_id)
        backend = self._backend_for_label(label)
        # summarize() outside the lock: it does I/O (zarr.open) and should
        # not hold the lock while blocking.
        summary = backend.summarize(dataset_id, fs_path)
        with self._lock:
            # Lazy-load before insert: ensures _cache contains all pre-existing
            # datasets, not just the newly registered one.
            self._ensure_loaded()
            self._cache[dataset_id] = summary  # type: ignore[index]

    def invalidate(self) -> None:
        """Mark the cache as dirty.

        The next call to list_datasets() or register_dataset() will trigger
        a full O(N) rescan of all backends.

        Does NOT destroy the StorageRegistry singleton. The @lru_cache on
        get_registry() is preserved — only the internal _cache dict is reset.
        """
        with self._lock:
            self._cache = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Populate _cache if it is None. Caller MUST hold _lock."""
        if self._cache is None:
            self._cache = {
                s.dataset_id: s
                for b in self._backends
                for s in b.list_datasets()
            }

    def _backend_for_label(self, label: str) -> StorageBackend:
        """Return the first backend that owns the given root label.

        Uses getattr(b, '_roots', None) to check label membership directly.
        When a second backend type is added, extend this with an explicit
        isinstance check or add a public roots() property to the Protocol.

        Raises:
            DatasetNotFound: if no registered backend owns label.
        """
        for b in self._backends:
            roots: dict | None = getattr(b, "_roots", None)
            if roots is not None and label in roots:
                return b
        registered = [b.name for b in self._backends]
        raise DatasetNotFound(
            f"No backend owns label {label!r}. "
            f"Registered backends: {registered}. "
            f"Check that ARRO_SERVER_DATA_ROOTS includes a root labelled {label!r}."
        )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

from functools import lru_cache  # noqa: E402  (import after class definition)


@lru_cache(maxsize=1)
def get_registry() -> StorageRegistry:
    """Return the process-wide StorageRegistry singleton.

    The singleton is preserved across cache invalidations — only the
    internal _cache dict is reset, not the object itself.
    Use ``get_registry().invalidate()`` to trigger a rescan.
    Use ``get_registry.cache_clear()`` only in test teardown for full reset.
    """
    settings = get_settings()
    backends: list[StorageBackend] = [ZarrFilesystemBackend(settings.resolved_roots)]
    return StorageRegistry(backends)


def reset_registry_cache() -> None:
    """Invalidate the registry cache without destroying the singleton.

    Backward-compatible replacement for the previous get_registry.cache_clear()
    call. Marks _cache = None so the next list_datasets() triggers a rescan.

    For test teardown requiring full singleton reset, call:
        get_registry.cache_clear()
    directly.
    """
    get_registry().invalidate()
