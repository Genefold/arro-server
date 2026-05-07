"""Storage abstraction.

Backends expose datasets identified by an opaque ``dataset_id`` of the form
``"<root_label>/<path>"``. Concrete backends (filesystem Zarr, future S3/GCS,
Parquet, etc.) implement :class:`StorageBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..slicing import ResolvedSlice


@dataclass(frozen=True)
class DatasetSummary:
    dataset_id: str
    root: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None = None
    kind: str = "array"  # "array" | "group" | "table"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetHandle:
    summary: DatasetSummary
    metadata: dict[str, Any]

    def read_window(self, rs: ResolvedSlice) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:  # pragma: no cover - default impl in backend
        return {}


@runtime_checkable
class StorageBackend(Protocol):
    name: str

    def list_datasets(self) -> list[DatasetSummary]: ...

    def open(self, dataset_id: str) -> DatasetHandle: ...
