from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from .. import __version__
from ..arrowspace_adapter import ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotFound, InvalidSlice
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
from .serializers import array_to_payload

router = APIRouter(prefix="/api")


def _registry() -> StorageRegistry:
    return get_registry()


def _arrowspace() -> ArrowSpaceAdapter:
    return load_arrowspace()


def _resolve_dataset_path(settings: Settings, dataset_id: str) -> Path:
    if "/" in dataset_id:
        label, rel = dataset_id.split("/", 1)
    else:
        label, rel = dataset_id, "."
    roots = settings.resolved_roots()
    root = roots.get(label)
    if root is None:
        raise DatasetNotFound(dataset_id)
    return root if rel in (".", "") else root / rel


# ---------------------------------------------------------------------------


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": load_arrowspace().backend,
        "data_roots": list(settings.resolved_roots().keys()),
    }


@router.get("/datasets")
def list_datasets(reg: StorageRegistry = Depends(_registry)) -> dict[str, Any]:
    items = reg.list_datasets()
    return {
        "count": len(items),
        "datasets": [
            {
                "id": s.dataset_id,
                "root": s.root,
                "path": s.path,
                "kind": s.kind,
                "shape": list(s.shape),
                "dtype": s.dtype,
                "chunks": list(s.chunks) if s.chunks else None,
            }
            for s in items
        ],
    }


@router.get("/datasets/{dataset_id:path}/metadata")
def dataset_metadata(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    return {
        "id": h.summary.dataset_id,
        "root": h.summary.root,
        "path": h.summary.path,
        "kind": h.summary.kind,
        "shape": list(h.summary.shape),
        "dtype": h.summary.dtype,
        "chunks": list(h.summary.chunks) if h.summary.chunks else None,
        "metadata": h.metadata,
    }


@router.get("/datasets/{dataset_id:path}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Row-oriented window over the leading axis. Suited to infinite scroll."""
    h = reg.open(dataset_id)
    if not h.summary.shape:
        raise InvalidSlice("dataset has no shape")
    eff_limit = limit or settings.default_window
    rs = parse_slice(None, h.summary.shape, offset=offset, limit=eff_limit)
    try:
        enforce_window_budget(rs, settings.max_window * max(1, _trailing_product(h.summary.shape)))
    except ValueError as e:
        raise InvalidSlice(str(e)) from e
    arr = h.read_window(rs)
    payload = array_to_payload(arr, preview_max_rows=eff_limit)
    total = h.summary.shape[0]
    next_offset = offset + payload["shape"][0] if payload["shape"] else offset
    return {
        "id": h.summary.dataset_id,
        "offset": offset,
        "limit": eff_limit,
        "total": total,
        "next_offset": next_offset if next_offset < total else None,
        "data": payload,
    }


@router.get("/datasets/{dataset_id:path}/slice")
def dataset_slice(
    dataset_id: str,
    spec: str = Query(..., alias="slice", description="Comma-separated per-axis slice spec"),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        rs = parse_slice(spec, h.summary.shape)
        enforce_window_budget(rs, settings.max_window * max(1, _trailing_product(h.summary.shape)))
    except ValueError as e:
        raise InvalidSlice(str(e)) from e
    arr = h.read_window(rs)
    return {
        "id": h.summary.dataset_id,
        "slice": spec,
        "out_shape": list(arr.shape),
        "data": array_to_payload(arr),
    }


@router.get("/datasets/{dataset_id:path}/manifold")
def dataset_manifold(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    path = _resolve_dataset_path(settings, dataset_id)
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "manifold": adapter.manifold(path),
    }


@router.get("/datasets/{dataset_id:path}/stats")
def dataset_stats(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    handle = reg.open(dataset_id)
    base = handle.stats()
    path = _resolve_dataset_path(settings, dataset_id)
    arrowspace_stats: dict[str, Any] | None = None
    try:
        arrowspace_stats = adapter.stats(path)
    except Exception as e:
        # Stats from ArrowSpace are best-effort; surface availability without 500.
        arrowspace_stats = {"unavailable": str(e)}
    return {
        "id": dataset_id,
        "basic": base,
        "arrowspace": arrowspace_stats,
        "backend": adapter.backend,
    }


@router.get("/datasets/{dataset_id:path}/search")
def dataset_search(
    dataset_id: str,
    q: str | None = Query(None, description="Free-text query"),
    limit: int = Query(20, ge=1, le=500),
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    path = _resolve_dataset_path(settings, dataset_id)
    return adapter.search(path, {"q": q, "limit": limit}) | {"id": dataset_id}


def _trailing_product(shape: tuple[int, ...]) -> int:
    if len(shape) <= 1:
        return 1
    p = 1
    for d in shape[1:]:
        p *= int(d)
    return p
