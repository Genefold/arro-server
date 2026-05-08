"""Filesystem-backed Zarr v3 storage backend.

Zarr is an optional dependency. Importing this module never fails; runtime
operations raise :class:`OptionalDependencyMissing` if zarr is unavailable.

Dataset IDs are URL-safe strings produced by :func:`make_dataset_id` —
filesystem slashes are encoded as ``--``.  Use :func:`decode_dataset_id`
to recover the original ``(label, path)`` pair for filesystem access.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetNotFound, OptionalDependencyMissing
from ..slicing import ResolvedSlice
from .base import DatasetHandle, DatasetSummary, decode_dataset_id, make_dataset_id

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
    """Convert a Zarr fill_value to a JSON-safe scalar.

    float NaN and Inf are not valid JSON; replace them with None so that
    metadata responses never trigger a serialization error.
    """
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    return v


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
    legacy ``.zarray`` / ``.zgroup``. Each discovered array is exposed as a
    dataset; groups are listed but not directly readable here.
    """

    name = "zarr-fs"

    def __init__(self, roots: dict[str, Path]):
        self._roots = roots

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
        return (
            (p / "zarr.json").exists()
            or (p / ".zarray").exists()
            or (p / ".zgroup").exists()
        )

    def _collect(
        self,
        label: str,
        root: Path,
        node: Any,
        rel: str,
    ) -> list[DatasetSummary]:
        out: list[DatasetSummary] = []
        # Use isinstance checks against the Zarr API rather than duck-typing.
        is_array = _ZARR_AVAILABLE and isinstance(node, zarr.Array)  # type: ignore[union-attr]
        if is_array:
            out.append(self._summarize_array(label, root, rel, node))
            return out
        # It's a group — emit the group + recurse arrays inside.
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
            attrs = dict(arr.attrs)
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
            attrs = dict(arr.attrs)
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
