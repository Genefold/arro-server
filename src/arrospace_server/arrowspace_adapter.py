"""Adapter for pyarrowspace / ArrowSpace metadata.

The package may not be available in every environment. We never import it at
module load — :func:`load` attempts the import lazily and returns a stub
adapter that raises :class:`OptionalDependencyMissing` on use.

ArrowSpace metadata is also looked up from sidecar files placed next to a
dataset. The convention used by this scaffold:

    <dataset_root>/_arrowspace/manifold.json
    <dataset_root>/_arrowspace/stats.json
    <dataset_root>/_arrowspace/index.<ext>     (opaque to this layer)

This means a dataset can advertise ArrowSpace metadata without the Python
package being installed, which is exactly what we want for boilerplate use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MetadataUnavailable, OptionalDependencyMissing

log = logging.getLogger(__name__)

SIDECAR_DIR = "_arrowspace"


@dataclass
class ArrowSpaceAdapter:
    available: bool
    backend: str  # "pyarrowspace" | "sidecar" | "none"

    def manifold(self, dataset_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def stats(self, dataset_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def search(self, dataset_path: Path, query: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class _SidecarAdapter(ArrowSpaceAdapter):
    def __init__(self) -> None:
        super().__init__(available=True, backend="sidecar")

    @staticmethod
    def _read(path: Path, name: str) -> dict[str, Any]:
        f = path / SIDECAR_DIR / name
        if not f.exists():
            raise MetadataUnavailable(f"{name} not present at {f}")
        try:
            return json.loads(f.read_text())
        except Exception as e:
            raise MetadataUnavailable(f"failed to parse {f}: {e}") from e

    def manifold(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "manifold.json")

    def stats(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "stats.json")

    def search(self, dataset_path: Path, query: dict[str, Any]) -> dict[str, Any]:
        # Sidecar mode: support naive linear lookup against an index.json
        # consisting of {"items": [{"id": ..., "tags": [...], "vector": [...]}]}.
        idx_file = dataset_path / SIDECAR_DIR / "index.json"
        if not idx_file.exists():
            raise MetadataUnavailable(
                f"search index missing: {idx_file}. Install pyarrowspace for vector search."
            )
        idx = json.loads(idx_file.read_text())
        items = idx.get("items", [])
        q = (query.get("q") or "").lower().strip()
        limit = int(query.get("limit") or 20)
        if not q:
            return {"backend": "sidecar", "results": items[:limit]}
        hits = [
            it
            for it in items
            if q in str(it.get("id", "")).lower()
            or any(q in str(t).lower() for t in it.get("tags", []))
        ]
        return {"backend": "sidecar", "results": hits[:limit], "query": q}


class _UnavailableAdapter(ArrowSpaceAdapter):
    def __init__(self) -> None:
        super().__init__(available=False, backend="none")

    def manifold(self, dataset_path: Path) -> dict[str, Any]:
        raise OptionalDependencyMissing("pyarrowspace", "manifold metadata")

    def stats(self, dataset_path: Path) -> dict[str, Any]:
        raise OptionalDependencyMissing("pyarrowspace", "ArrowSpace statistics")

    def search(self, dataset_path: Path, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("pyarrowspace", "ArrowSpace search")


class _PyArrowSpaceAdapter(ArrowSpaceAdapter):  # pragma: no cover - depends on optional pkg
    def __init__(self, module: Any) -> None:
        super().__init__(available=True, backend="pyarrowspace")
        self._mod = module

    def _load(self, dataset_path: Path) -> Any:
        # The exact pyarrowspace API is not pinned here. We try a few common
        # entry points and fall through to the sidecar adapter on failure.
        for name in ("open_dataset", "open", "Dataset"):
            fn = getattr(self._mod, name, None)
            if callable(fn):
                return fn(str(dataset_path))
        raise OptionalDependencyMissing("pyarrowspace", "no compatible open() entry point")

    def manifold(self, dataset_path: Path) -> dict[str, Any]:
        ds = self._load(dataset_path)
        for name in ("manifold", "get_manifold"):
            fn = getattr(ds, name, None)
            if callable(fn):
                return _to_jsonable(fn())
        raise MetadataUnavailable("pyarrowspace dataset exposes no manifold()")

    def stats(self, dataset_path: Path) -> dict[str, Any]:
        ds = self._load(dataset_path)
        for name in ("stats", "summary", "describe"):
            fn = getattr(ds, name, None)
            if callable(fn):
                return _to_jsonable(fn())
        raise MetadataUnavailable("pyarrowspace dataset exposes no stats()")

    def search(self, dataset_path: Path, query: dict[str, Any]) -> dict[str, Any]:
        ds = self._load(dataset_path)
        fn = getattr(ds, "search", None)
        if callable(fn):
            return _to_jsonable(fn(**query))
        raise MetadataUnavailable("pyarrowspace dataset exposes no search()")


def _to_jsonable(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


_cached: ArrowSpaceAdapter | None = None


def load() -> ArrowSpaceAdapter:
    """Pick the best available adapter.

    Order of preference:
        1. pyarrowspace if importable
        2. Sidecar JSON files under ``_arrowspace/`` (always works)
    """
    global _cached
    if _cached is not None:
        return _cached
    try:
        import pyarrowspace  # type: ignore

        _cached = _PyArrowSpaceAdapter(pyarrowspace)
        log.info("ArrowSpace adapter: pyarrowspace")
        return _cached
    except Exception as e:
        log.info("pyarrowspace unavailable (%s); falling back to sidecar adapter", e)
    _cached = _SidecarAdapter()
    return _cached


def reset_adapter_cache() -> None:
    global _cached
    _cached = None
