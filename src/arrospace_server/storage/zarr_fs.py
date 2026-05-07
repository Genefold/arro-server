"""Filesystem-backed Zarr v3 storage backend.

Zarr is an optional dependency. Importing this module never fails; runtime
operations raise :class:`OptionalDependencyMissing` if zarr is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetNotFound, OptionalDependencyMissing
from ..slicing import ResolvedSlice
from .base import DatasetHandle, DatasetSummary

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


class _ZarrArrayHandle(DatasetHandle):
    def __init__(self, summary: DatasetSummary, metadata: dict[str, Any], arr: Any):
        super().__init__(summary=summary, metadata=metadata)
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
        # Top-level: try opening the root itself first.
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
        # Otherwise, treat root as a directory of datasets.
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
        # Heuristic: is this an array?
        if hasattr(node, "shape") and hasattr(node, "dtype") and not hasattr(node, "arrays"):
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
                dataset_id=f"{label}/{rel}".rstrip("/."),
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
        ds_id = f"{label}/{rel_clean}" if rel_clean != "." else label
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
        if "/" in dataset_id:
            label, rel = dataset_id.split("/", 1)
        else:
            label, rel = dataset_id, "."
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
        if not (hasattr(arr, "shape") and hasattr(arr, "dtype")):
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
            "fill_value": getattr(arr, "fill_value", None),
            "order": getattr(arr, "order", None),
        }
        # Coerce fill_value to JSON-friendly.
        fv = metadata["fill_value"]
        if isinstance(fv, np.generic):
            metadata["fill_value"] = fv.item()
        return _ZarrArrayHandle(summary=summary, metadata=metadata, arr=arr)
