"""Filesystem-backed Zarr v3 storage backend.

Zarr is an optional dependency.  Importing this module never fails; runtime
operations raise :class:`OptionalDependencyMissing` if zarr is unavailable.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from ..api.serializers import deep_sanitize
from ..errors import DatasetNotFound, OptionalDependencyMissing, VectorDtypeMismatch, VectorShapeMismatch
from ..slicing import ResolvedSlice
from .base import DatasetHandle, DatasetSummary, decode_dataset_id, make_dataset_id

if TYPE_CHECKING:
    from .registry import StorageRegistry

log = logging.getLogger(__name__)

try:  # pragma: no cover - import-time guard
    import zarr  # type: ignore

    _ZARR_AVAILABLE = True
except Exception:
    zarr = None  # type: ignore[assignment]
    _ZARR_AVAILABLE = False


def zarr_available() -> bool:
    return _ZARR_AVAILABLE


def _require_zarr() -> None:
    if not _ZARR_AVAILABLE:
        raise OptionalDependencyMissing("zarr", "Zarr filesystem backend")


def _safe_fill_value(v: Any) -> Any:
    """Convert a Zarr fill_value to a JSON-safe Python scalar.

    Handles:
    - ``float`` NaN / Inf  -> ``None``
    - ``numpy.floating``   -> ``float`` (with NaN/Inf -> ``None``)
    - ``numpy.integer``    -> ``int``
    - ``numpy.bool_``      -> ``bool``
    - ``complex``          -> ``{"re": ..., "im": ...}``
    - Any other numpy type -> coerced via ``deep_sanitize``
    """
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, np.floating):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    # Catch-all: run through deep_sanitize for structured / exotic types.
    return deep_sanitize(v)


class _ZarrArrayHandle(DatasetHandle):
    def __init__(
        self,
        summary: DatasetSummary,
        metadata: dict[str, Any],
        arr: Any,
        fs_path: Path,
    ):
        super().__init__(summary=summary, metadata=metadata, fs_path=fs_path)
        self._arr = arr

    def read_window(self, rs: ResolvedSlice) -> np.ndarray:
        data = self._arr[rs.selectors]
        return np.ascontiguousarray(data)

    def stats(self) -> dict[str, Any]:
        s: dict[str, Any] = {
            "shape": list(self.summary.shape),
            "dtype": self.summary.dtype,
            "chunks": list(self.summary.chunks) if self.summary.chunks else None,
            "size": int(np.prod(self.summary.shape)) if self.summary.shape else 0,
        }
        return s


class ZarrFilesystemBackend:
    """Walks configured roots looking for ``zarr.json`` markers (v3) or
    legacy ``.zarray`` / ``.zgroup``.  Each discovered array is exposed as a
    dataset; groups are listed but not directly readable.
    """

    name = "zarr-fs"

    def __init__(self, roots: dict[str, Path]):
        self._roots = roots
        self._write_locks: dict[str, threading.Lock] = {}
        self._write_locks_mutex = threading.Lock()

    # ----- discovery ---------------------------------------------------

    def list_datasets(self) -> list[DatasetSummary]:
        if not _ZARR_AVAILABLE:
            return []
        out: list[DatasetSummary] = []
        for label, root in self._roots.items():
            if not root.exists():
                log.warning("data root %s does not exist: %s", label, root)
                continue
            out.extend(self._scan_root(label, root))
        return out

    def _scan_root(self, label: str, root: Path) -> list[DatasetSummary]:
        found: list[DatasetSummary] = []
        rel = "."
        if self._is_zarr_node(root):
            try:
                node = zarr.open(str(root), mode="r")  # type: ignore[union-attr]
            except Exception as e:
                log.warning("failed to open root zarr at %s: %s", root, e)
                node = None
            if node is not None:
                found.extend(self._collect(label, root, node, rel))
            return found
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if self._is_zarr_node(child):
                try:
                    node = zarr.open(str(child), mode="r")  # type: ignore[union-attr]
                except Exception as e:
                    log.warning("failed to open zarr at %s: %s", child, e)
                    continue
                found.extend(self._collect(label, root, node, child.name))
        return found

    @staticmethod
    def _is_zarr_node(p: Path) -> bool:
        return (p / "zarr.json").exists() or (p / ".zarray").exists() or (p / ".zgroup").exists()

    def _collect(
        self,
        label: str,
        root: Path,
        node: Any,
        rel: str,
    ) -> list[DatasetSummary]:
        out: list[DatasetSummary] = []
        is_array = _ZARR_AVAILABLE and isinstance(node, zarr.Array)  # type: ignore[union-attr]
        if is_array:
            out.append(self._summarize_array(label, root, rel, node))
            return out
        try:
            arrays = dict(node.arrays())
        except Exception:
            arrays = {}
        try:
            groups = dict(node.groups())
        except Exception:
            groups = {}
        out.append(
            DatasetSummary(
                dataset_id=make_dataset_id(label, rel),
                root=label,
                path=rel,
                shape=(),
                dtype="",
                chunks=None,
                kind="group",
                extra={"n_arrays": len(arrays), "n_groups": len(groups)},
            )
        )
        for name, arr in arrays.items():
            sub = f"{rel}/{name}".lstrip("./")
            out.append(self._summarize_array(label, root, sub, arr))
        for name, sub_grp in groups.items():
            sub = f"{rel}/{name}".lstrip("./")
            out.extend(self._collect(label, root, sub_grp, sub))
        return out

    @staticmethod
    def _summarize_array(label: str, root: Path, rel: str, arr: Any) -> DatasetSummary:
        rel_clean = rel.lstrip("./") or "."
        ds_id = make_dataset_id(label, rel_clean)
        try:
            attrs = deep_sanitize(dict(arr.attrs))
        except Exception:
            attrs = {}
        return DatasetSummary(
            dataset_id=ds_id,
            root=label,
            path=rel_clean,
            shape=tuple(int(x) for x in arr.shape),
            dtype=str(arr.dtype),
            chunks=tuple(int(x) for x in arr.chunks) if getattr(arr, "chunks", None) else None,
            kind="array",
            extra={"attrs": attrs},
        )

    # ----- summarize ---------------------------------------------------

    def summarize(self, dataset_id: str, fs_path: Path) -> DatasetSummary:
        """Return a DatasetSummary for the single Zarr node at fs_path.

        Opens exactly one zarr node (one zarr.open() call) and delegates to
        _summarize_array. This is the O(1) path used by
        StorageRegistry.register_dataset() after POST /upload/commit.

        The cost is identical to what _scan_root pays per node during a full
        scan, but scoped to a single known path — no directory walk, no
        iterdir(), no recursive _collect().

        Args:
            dataset_id: URL-safe dataset ID used as the summary's dataset_id.
                        Must be pre-computed by the caller (upload handler).
            fs_path:    Absolute path to the Zarr array or group root directory.

        Returns:
            DatasetSummary with kind='array'.

        Raises:
            DatasetNotFound: if zarr is unavailable, fs_path does not exist,
                             or the node at fs_path is a group (not an array).
        """
        _require_zarr()
        label, rel = decode_dataset_id(dataset_id)
        if not fs_path.exists():
            raise DatasetNotFound(dataset_id)
        try:
            arr = zarr.open(str(fs_path), mode="r")  # type: ignore[union-attr]
        except Exception as exc:
            raise DatasetNotFound(f"{dataset_id} ({exc})") from exc
        if not isinstance(arr, zarr.Array):  # type: ignore[union-attr]
            raise DatasetNotFound(f"{dataset_id} is a group, not an array")
        return self._summarize_array(label, fs_path.parent, rel, arr)

    # ----- open --------------------------------------------------------

    def open(self, dataset_id: str) -> DatasetHandle:
        _require_zarr()
        label, rel = decode_dataset_id(dataset_id)
        root = self._roots.get(label)
        if root is None:
            raise DatasetNotFound(dataset_id)
        target = root if rel in (".", "") else root / rel
        if not target.exists():
            raise DatasetNotFound(dataset_id)
        try:
            arr = zarr.open(str(target), mode="r")  # type: ignore[union-attr]
        except Exception as e:
            raise DatasetNotFound(f"{dataset_id} ({e})") from e
        if not isinstance(arr, zarr.Array):  # type: ignore[union-attr]
            raise DatasetNotFound(f"{dataset_id} is a group, not an array")
        summary = self._summarize_array(label, root, rel, arr)
        try:
            # deep_sanitize ensures no numpy scalars survive into the metadata
            # dict, which would cause PydanticSerializationError at response time.
            attrs = deep_sanitize(dict(arr.attrs))
        except Exception:
            attrs = {}
        metadata = {
            "shape": list(summary.shape),
            "dtype": summary.dtype,
            "chunks": list(summary.chunks) if summary.chunks else None,
            "attrs": attrs,
            "fill_value": _safe_fill_value(getattr(arr, "fill_value", None)),
            "order": getattr(arr, "order", None),
        }
        return _ZarrArrayHandle(summary=summary, metadata=metadata, arr=arr, fs_path=target)

    # ----- write --------------------------------------------------------

    def _get_write_lock(self, dataset_id: str) -> threading.Lock:
        """Return (creating if necessary) the per-dataset write lock.

        Two-phase commit operations (append, overwrite) on the same dataset
        must be serialized so that resize + write + cache-update is atomic.
        Datasets do not share locks — concurrent writes on different datasets
        proceed in parallel.
        """
        with self._write_locks_mutex:
            if dataset_id not in self._write_locks:
                self._write_locks[dataset_id] = threading.Lock()
            return self._write_locks[dataset_id]

    def append_vectors(
        self,
        dataset_id: str,
        vecs: np.ndarray,
        registry: StorageRegistry | None = None,
    ) -> tuple[int, int]:
        """Append M new row-vectors to an existing 2-D Zarr array.

        The array at dataset_id is opened in ``"r+"`` mode, resized to
        accommodate the new rows, and the new vectors are written to the
        newly allocated slice.

        Complexity:
            - Path resolution & open: O(1)  — single ``zarr.json`` read.
            - Validation:             O(1)  — shape/dtype from metadata.
            - Resize:                 O(1)  — Zarr v3 metadata-only.
            - Write:                  O(M·D) — only the M new rows.
            - Cache update:           O(1)  — ``register_dataset()``.

        Args:
            dataset_id: URL-safe dataset ID. Must exist and be a 2-D array.
            vecs:       NumPy array of shape (M, D), M > 0. D and dtype must
                        match the existing array.
            registry:   Optional ``StorageRegistry``. If provided,
                        ``register_dataset()`` is called inside the per-dataset
                        write lock to update the cached shape before the lock
                        is released.

        Returns:
            Tuple ``(start_row, new_nrows)`` where:
              - ``start_row`` — row index of the first appended vector
                (= old row count).
              - ``new_nrows`` — total row count after append (= old_n + M).

        Raises:
            DatasetNotFound:       dataset_id does not resolve to an array.
            VectorShapeMismatch:   vecs.ndim != 2, D mismatch, or M == 0.
            VectorDtypeMismatch:   vecs.dtype incompatible with arr.dtype.
            OptionalDependencyMissing: zarr package not installed.
        """
        _require_zarr()

        label, rel = decode_dataset_id(dataset_id)
        root = self._roots.get(label)
        if root is None:
            raise DatasetNotFound(dataset_id)
        fs_path = root if rel in (".", "") else root / rel
        if not fs_path.exists():
            raise DatasetNotFound(dataset_id)

        # Open once in "r" mode for validation (outside the write lock).
        arr = zarr.open(str(fs_path), mode="r")
        if not isinstance(arr, zarr.Array):
            raise DatasetNotFound(f"{dataset_id} is a group, not an array")

        if vecs.ndim != 2:
            raise VectorShapeMismatch(
                f"Expected 2-D array, got {vecs.ndim}D for '{dataset_id}'"
            )
        if vecs.shape[0] == 0:
            raise VectorShapeMismatch(
                f"Cannot append zero vectors to '{dataset_id}'"
            )
        expected_d = arr.shape[1]
        if vecs.shape[1] != expected_d:
            raise VectorShapeMismatch(
                f"Dimension mismatch for '{dataset_id}': "
                f"vectors have {vecs.shape[1]} features, "
                f"dataset has {expected_d}"
            )
        if vecs.dtype != arr.dtype:
            raise VectorDtypeMismatch(
                f"dtype mismatch for '{dataset_id}': "
                f"vectors are {vecs.dtype}, dataset is {arr.dtype}"
            )

        # Write phase inside the per-dataset lock.
        with self._get_write_lock(dataset_id):
            arr = zarr.open(str(fs_path), mode="r+")
            old_n = int(arr.shape[0])
            M = int(vecs.shape[0])
            D = int(arr.shape[1])
            new_n = old_n + M
            arr.resize((new_n, D))
            arr[old_n:new_n, :] = vecs
            if registry is not None:
                registry.register_dataset(dataset_id, fs_path)
            return old_n, new_n
