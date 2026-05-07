from __future__ import annotations

from functools import lru_cache

from ..errors import DatasetNotFound
from ..settings import get_settings
from .base import DatasetHandle, DatasetSummary, StorageBackend
from .zarr_fs import ZarrFilesystemBackend


class StorageRegistry:
    """Multiplexes across registered backends.

    For now we ship one backend (filesystem Zarr v3). Object-store backends
    can register themselves here without touching the API layer.
    """

    def __init__(self, backends: list[StorageBackend]):
        self._backends = backends

    def list_datasets(self) -> list[DatasetSummary]:
        out: list[DatasetSummary] = []
        for b in self._backends:
            out.extend(b.list_datasets())
        return out

    def open(self, dataset_id: str) -> DatasetHandle:
        last_err: Exception | None = None
        for b in self._backends:
            try:
                return b.open(dataset_id)
            except DatasetNotFound as e:
                last_err = e
                continue
        raise last_err or DatasetNotFound(dataset_id)


@lru_cache(maxsize=1)
def get_registry() -> StorageRegistry:
    settings = get_settings()
    backends: list[StorageBackend] = [ZarrFilesystemBackend(settings.resolved_roots())]
    return StorageRegistry(backends)


def reset_registry_cache() -> None:
    get_registry.cache_clear()
