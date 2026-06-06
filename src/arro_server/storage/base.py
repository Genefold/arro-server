"""Storage abstraction.

Backends expose datasets via a URL-safe ``dataset_id`` of the form
``"<root_label>--<path_segment>--<...>"`` (slashes replaced with ``--``).
The human-readable ``root`` and ``path`` fields are preserved separately
in :class:`DatasetSummary` for display purposes.

Concrete backends (filesystem Zarr, future S3/GCS, Parquet, etc.) implement
:class:`StorageBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..slicing import ResolvedSlice

# ---------------------------------------------------------------------------
# Dataset ID encoding helpers
# ---------------------------------------------------------------------------

_SEP = "--"  # URL-safe separator between label and path components


def make_dataset_id(label: str, path: str) -> str:
    """Encode a ``(label, path)`` pair into a URL-safe dataset ID.

    Filesystem slashes and backslashes are replaced with ``--`` so the
    resulting ID contains no characters that conflict with HTTP path
    segment boundaries.

    Examples::

        make_dataset_id("main", "cube")        -> "main--cube"
        make_dataset_id("main", "sub/array")   -> "main--sub--array"
        make_dataset_id("main", ".")           -> "main"
        make_dataset_id("main", "")            -> "main"
    """
    clean = path.strip("./").replace("\\", "/") if path else ""
    if not clean:
        return label
    parts = [label] + [p for p in clean.split("/") if p]
    return _SEP.join(parts)


def decode_dataset_id(dataset_id: str) -> tuple[str, str]:
    """Decode a URL-safe dataset ID back to ``(label, rel_path)``.

    Examples::

        decode_dataset_id("main--cube")           -> ("main", "cube")
        decode_dataset_id("main--sub--array")     -> ("main", "sub/array")
        decode_dataset_id("main")                 -> ("main", ".")
    """
    parts = dataset_id.split(_SEP)
    label = parts[0]
    rel = "/".join(parts[1:]) if len(parts) > 1 else "."
    return label, rel


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


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
    # Filesystem path to the dataset root directory.  Set by filesystem
    # backends so that sidecar readers can locate _arrowspace/ files without
    # duplicating path-resolution logic in the route layer.
    fs_path: Path | None = None

    def read_window(self, rs: ResolvedSlice) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:  # pragma: no cover
        return {}


@runtime_checkable
class StorageBackend(Protocol):
    name: str

    def list_datasets(self) -> list[DatasetSummary]: ...

    def open(self, dataset_id: str) -> DatasetHandle: ...

    def summarize(self, dataset_id: str, fs_path: Path) -> DatasetSummary:
        """Return a DatasetSummary for a single dataset at fs_path.

        Called by StorageRegistry.register_dataset() for O(1) post-upload
        registration. Implementors must open only the single node at fs_path
        — equivalent to what _scan_root does per node, but scoped to one path.

        Args:
            dataset_id: URL-safe dataset ID (e.g. "main--cube").
            fs_path:    Absolute filesystem path to the Zarr node root dir.

        Returns:
            DatasetSummary for the node at fs_path.

        Raises:
            DatasetNotFound: if fs_path does not contain a valid dataset.
        """
        ...


