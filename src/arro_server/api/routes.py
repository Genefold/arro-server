"""API route handlers for arro-server.

All routes are mounted under the /api prefix.
Dataset IDs use '--' as the root/path separator (e.g. 'main--matrix').

Endpoint map
-----------
GET  /api/health
GET  /api/datasets
GET  /api/datasets/{id}/metadata
GET  /api/datasets/{id}/data
GET  /api/datasets/{id}/slice
GET  /api/datasets/{id}/stats
GET  /api/datasets/{id}/manifold
GET  /api/datasets/{id}/search               -- keyword (sidecar)
POST /api/datasets/{id}/index                -- build index
DELETE /api/datasets/{id}/index              -- delete index + purge files
GET  /api/datasets/{id}/lambdas              -- eigenvalue distribution
GET  /api/datasets/{id}/graph_laplacian      -- GL metadata
GET  /api/datasets/{id}/items                -- all items from index
GET  /api/datasets/{id}/items/{n}            -- single item
POST /api/datasets/{id}/search               -- unified spectral vector search with mode selector
POST /api/datasets/{id}/search/energy        -- energy vector search
POST /api/datasets/{id}/search/hybrid        -- hybrid vector search
POST /api/datasets/{id}/search/linear        -- linear sorted search
POST /api/datasets/{id}/search/batch         -- batch vector search
GET  /api/datasets/{id}/spot/motives/eigen
GET  /api/datasets/{id}/spot/motives/energy
GET  /api/datasets/{id}/spot/subgraphs/centroids
GET  /api/datasets/{id}/spot/subgraphs/motives
GET  /api/datasets/{id}/graph?fmt=csr|dense                -- Laplacian matrix export
GET  /api/datasets/{id}/spectral_metrics                   -- full spectral analytics
POST /api/admin/reload                                     -- hot-reload StorageRegistry + ArrowSpaceAdapter cache
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .. import __version__
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS, ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotSliceable, InvalidSlice, OptionalDependencyMissing
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice, trailing_product
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
from .schemas import (
    IndexBuildRequest,
    SearchBatchRequest,
    SearchEnergyRequest,
    SearchHybridRequest,
    SearchLinearRequest,
    SearchModeRequest,
)
from .serializers import array_to_payload

router = APIRouter(prefix="/api")
admin_router = APIRouter(prefix="/api/admin")


def _registry() -> StorageRegistry:
    return get_registry()


def _arrowspace() -> ArrowSpaceAdapter:
    return load_arrowspace()


# ---------------------------------------------------------------------------
# Admin — reload / cache invalidation
# ---------------------------------------------------------------------------


def _require_admin_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Dependency — raises 401 if admin_token is set and header doesn't match."""
    if settings.admin_token is None:
        return
    expected = f"Bearer {settings.admin_token}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header.",
        )


@admin_router.post("/reload", dependencies=[Depends(_require_admin_token)])
def admin_reload(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Invalidate all LRU caches and re-scan data_roots.

    Call this after writing new Zarr datasets to a data_root from an
    external process (e.g. arro-memory on a shared volume).
    """
    from ..arrowspace_adapter import load as load_adapter
    from ..arrowspace_adapter import reset_adapter_cache
    from ..storage.registry import get_registry, reset_registry_cache

    reset_registry_cache()        # marks _cache = None (invalidate, not singleton destroy)
    reset_adapter_cache()

    registry = get_registry()
    datasets = registry.list_datasets()  # triggers full O(N) rescan here (expected for admin)

    new_adapter = load_adapter()
    index_store = Path(settings.index_store).expanduser().resolve()
    try:
        loaded = new_adapter.reload_from_manifest(index_store)
    except Exception:
        loaded = []

    return {
        "reloaded": True,
        "datasets_found": len([d for d in datasets if d.kind == "array"]),
        "data_roots": list(settings.resolved_roots.keys()),
        "indexed_datasets": loaded,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    adapter = load_arrowspace()
    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": adapter.backend,
        "arrowspace_available": adapter.available,
        "data_roots": list(settings.resolved_roots.keys()),
        "indexed_datasets": adapter.indexed_datasets(),
    }


# ---------------------------------------------------------------------------
# Dataset discovery + raw Zarr access
# ---------------------------------------------------------------------------


@router.get("/datasets")
def list_datasets(reg: StorageRegistry = Depends(_registry)) -> dict[str, Any]:
    entries = reg.list_datasets()
    return {
        "datasets": [
            {
                "id": s.dataset_id,
                "kind": "array",
                "root": s.root,
                "path": s.path,
                "shape": list(s.shape),
                "dtype": s.dtype,
            }
            for s in entries
        ]
    }


@router.get("/datasets/{dataset_id:path}/metadata")
def dataset_metadata(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    s = h.summary
    return {
        "id": dataset_id,
        "shape": list(s.shape),
        "dtype": s.dtype,
        "chunks": list(s.chunks) if s.chunks else None,
        "attrs": s.extra,
    }


@router.get("/datasets/{dataset_id:path}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    effective_limit = limit if limit is not None else settings.default_window
    rs = parse_slice(None, h.summary.shape, offset=offset, limit=effective_limit)
    try:
        # Window budget is scaled by the product of trailing dimensions so that
        # multi-dimensional arrays are bounded by total element count, not rows.
        enforce_window_budget(rs, settings.max_window * max(1, trailing_product(h.summary.shape)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    arr = h.read_window(rs)
    next_off = rs.selectors[0].stop if isinstance(rs.selectors[0], slice) else None
    if next_off is not None and next_off >= h.summary.shape[0]:
        next_off = None
    return {
        "id": dataset_id,
        "offset": offset,
        "limit": effective_limit,
        "next_offset": next_off,
        "data": array_to_payload(arr),
    }


@router.get("/datasets/{dataset_id:path}/slice")
def dataset_slice(
    dataset_id: str,
    spec: str = Query(..., alias="slice"),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        rs = parse_slice(spec, h.summary.shape)
        enforce_window_budget(rs, settings.max_window * max(1, trailing_product(h.summary.shape)))
    except InvalidSlice as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatasetNotSliceable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    arr = h.read_window(rs)
    return {
        "id": dataset_id,
        "slice": spec,
        "out_shape": list(arr.shape),
        "data": array_to_payload(arr),
    }


# ---------------------------------------------------------------------------
# ArrowSpace sidecar endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id:path}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    # Use has_index() instead of reaching into _cache directly (encapsulation)
    if adapter.has_index(dataset_id):
        stats = adapter.stats_data(dataset_id)  # type: ignore[attr-defined]
        return {
            "id": dataset_id,
            "backend": adapter.backend,
            "arrowspace_available": adapter.available,
            "stats": stats,
        }
    try:
        data = adapter.sidecar_stats(h.fs_path)
        s = h.summary
        data.setdefault("shape", list(s.shape))
        data.setdefault("dtype", s.dtype)
        return {
            "id": dataset_id,
            "backend": "sidecar",
            "arrowspace_available": adapter.available,
            "stats": data,
        }
    except Exception:
        s = h.summary
        return {
            "id": dataset_id,
            "backend": "none",
            "arrowspace_available": adapter.available,
            "stats": {"shape": list(s.shape), "dtype": s.dtype},
        }


@router.get("/datasets/{dataset_id:path}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    # Use has_index() instead of reaching into _cache directly (encapsulation)
    if adapter.has_index(dataset_id):
        data = adapter.manifold_data(dataset_id)  # type: ignore[attr-defined]
        return {
            "id": dataset_id,
            "backend": adapter.backend,
            "arrowspace_available": adapter.available,
            "source": "live",
            "manifold": data,
        }
    try:
        data = adapter.sidecar_manifold(h.fs_path)
        return {
            "id": dataset_id,
            "backend": "sidecar",
            "arrowspace_available": adapter.available,
            "source": "sidecar",
            "manifold": data,
        }
    except Exception:
        return {
            "id": dataset_id,
            "backend": "none",
            "arrowspace_available": adapter.available,
            "source": "unavailable",
            "manifold": None,
        }


@router.get("/datasets/{dataset_id:path}/search")
def dataset_search_sidecar(
    dataset_id: str,
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    results = adapter.sidecar_search(h.fs_path, q, limit=limit)
    return {"id": dataset_id, "query": q, "results": results}


# ---------------------------------------------------------------------------
# ArrowSpace live index
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id:path}/index")
def build_index(
    dataset_id: str,
    body: IndexBuildRequest = IndexBuildRequest(),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Build (or rebuild) the ArrowSpace graph-Laplacian index."""
    h = reg.open(dataset_id)
    rs = parse_slice(None, h.summary.shape, offset=0, limit=h.summary.shape[0])
    arr = h.read_window(rs)
    index_store = Path(settings.index_store).expanduser().resolve()
    effective_params = body.graph_params or DEFAULT_GRAPH_PARAMS
    try:
        meta = adapter.build_index(
            dataset_id=dataset_id,
            array=arr,
            index_store=index_store,
            graph_params=effective_params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "id": dataset_id,
        "built": True,
        "graph_params": effective_params,
        "nitems": meta["nitems"],
        "nfeatures": meta["nfeatures"],
        "nclusters": meta["nclusters"],
    }


@router.get("/datasets/{dataset_id:path}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.lambdas(dataset_id)
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        **data,
    }


@router.get("/datasets/{dataset_id:path}/graph_laplacian")
def dataset_graph_laplacian(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.graph_laplacian_info(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id:path}/items")
def dataset_get_all_items(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.get_all_items(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id:path}/items/{item_index}")
def dataset_get_item(
    dataset_id: str,
    item_index: int,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.get_item(dataset_id, item_index)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.post("/datasets/{dataset_id:path}/search/energy")
def dataset_search_energy(
    dataset_id: str,
    body: SearchEnergyRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_energy(dataset_id, body.model_dump())
    return {"id": dataset_id, **data}


@router.post("/datasets/{dataset_id:path}/search/hybrid")
def dataset_search_hybrid(
    dataset_id: str,
    body: SearchHybridRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_hybrid(dataset_id, body.model_dump())
    return {"id": dataset_id, **data}


@router.post("/datasets/{dataset_id:path}/search/linear")
def dataset_search_linear(
    dataset_id: str,
    body: SearchLinearRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_linear_sorted(dataset_id, body.model_dump())
    return {"id": dataset_id, **data}


@router.post("/datasets/{dataset_id:path}/search/batch")
def dataset_search_batch(
    dataset_id: str,
    body: SearchBatchRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    try:
        data = adapter.search_batch(dataset_id, body.model_dump())
        return {"id": dataset_id, **data}
    except OptionalDependencyMissing as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id:path}/spot/motives/eigen")
def dataset_spot_motives_eigen(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_motives_eigen(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id:path}/spot/motives/energy")
def dataset_spot_motives_energy(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_motives_energy(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id:path}/spot/subgraphs/centroids")
def dataset_spot_subg_centroids(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_subg_centroids(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id:path}/spot/subgraphs/motives")
def dataset_spot_subg_motives(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_subg_motives(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


# ---------------------------------------------------------------------------
# Phase 3 — Analytics endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id:path}/graph")
def dataset_graph(
    dataset_id: str,
    fmt: str = Query("csr", description="'csr' | 'dense'"),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if fmt not in ("csr", "dense"):
        raise HTTPException(status_code=422, detail="fmt must be 'csr' or 'dense'")
    try:
        result = adapter.graph_export(dataset_id, fmt)
        if fmt == "dense":
            nnodes = result.get("nnodes", 0)
            if nnodes > settings.max_window:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Dense export requires {nnodes * nnodes:,} floats "
                        f"(nnodes={nnodes} exceeds max_window={settings.max_window}). "
                        "Use fmt=csr instead."
                    ),
                )
        return result
    except OptionalDependencyMissing as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id:path}/spectral_metrics")
def dataset_spectral_metrics(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    try:
        return adapter.spectral_metrics(dataset_id)
    except OptionalDependencyMissing as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("/datasets/{dataset_id:path}/search")
def dataset_search(
    dataset_id: str,
    body: SearchModeRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    q_dict = {
        "vector": body.vector,
        "mode": body.mode,
        "tau": body.tau,
        "alpha": body.alpha,
        "k": body.k,
    }
    try:
        result = adapter.search_with_mode(dataset_id, q_dict)
        return {"id": dataset_id, "arrowspace_available": adapter.available, **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OptionalDependencyMissing:
        raise HTTPException(
            status_code=501, detail="search_with_mode requires arrowspace"
        ) from None


# ---------------------------------------------------------------------------
# Index lifecycle — Phase 2
# ---------------------------------------------------------------------------


@router.delete("/datasets/{dataset_id:path}/index")
def delete_index(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Delete a built index from the LRU cache and from disk.

    Removes:
    - The entry from the in-memory LRU cache
    - The ArrowSpace Parquet files written by the builder
    - The CSR Zarr directory
    - The entry from index_manifest.json

    Returns 404 if no index exists for the given dataset ID.
    """
    index_store = Path(settings.index_store).expanduser().resolve()
    existed = adapter.delete_index(dataset_id=dataset_id, index_store=index_store)
    if not existed:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for dataset '{dataset_id}'.",
        )
    return {"id": dataset_id, "deleted": True}
