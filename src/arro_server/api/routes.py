"""API route handlers for arro-server.

All routes are mounted under the /api prefix.
Dataset IDs use '--' as the root/path separator (e.g. 'main--matrix').

Endpoint map
------------
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
POST /api/datasets/{id}/search               -- spectral vector search
POST /api/datasets/{id}/search/energy        -- energy vector search
POST /api/datasets/{id}/search/hybrid        -- hybrid vector search
POST /api/datasets/{id}/search/linear        -- linear sorted search
POST /api/datasets/{id}/search/batch         -- batch vector search
GET  /api/datasets/{id}/spot/motives/eigen
GET  /api/datasets/{id}/spot/motives/energy
GET  /api/datasets/{id}/spot/subgraphs/centroids
GET  /api/datasets/{id}/spot/subgraphs/motives
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import __version__
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS, ArrowSpaceAdapter, _ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotSliceable, InvalidSlice
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
    SearchRequest,
)
from .serializers import array_to_payload

router = APIRouter(prefix="/api")


def _registry() -> StorageRegistry:
    return get_registry()


def _arrowspace() -> ArrowSpaceAdapter:
    return load_arrowspace()


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
        # Phase 2: list dataset IDs currently loaded in the LRU index cache
        "indexed_datasets": adapter.indexed_datasets(),
    }


# ---------------------------------------------------------------------------
# Dataset discovery + raw Zarr access
# ---------------------------------------------------------------------------


@router.get("/datasets")
def list_datasets(reg: StorageRegistry = Depends(_registry)) -> dict[str, Any]:
    return {"datasets": reg.list()}


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
        "attrs": s.attrs,
    }


@router.get("/datasets/{dataset_id:path}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=None),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    effective_limit = limit if limit is not None else settings.default_window
    rs = parse_slice(None, h.summary.shape, offset=offset, limit=effective_limit)
    rs = enforce_window_budget(rs, h.summary.shape, settings.max_window)
    arr = h.read_window(rs)
    next_off = rs.offset + rs.limit
    if next_off >= h.summary.shape[0]:
        next_off = None
    return {
        "id": dataset_id,
        "offset": rs.offset,
        "limit": rs.limit,
        "next_offset": next_off,
        "data": array_to_payload(arr),
    }


@router.get("/datasets/{dataset_id:path}/slice")
def dataset_slice(
    dataset_id: str,
    slice: str = Query(...),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        rs = parse_slice(slice, h.summary.shape)
    except InvalidSlice as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DatasetNotSliceable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rs = enforce_window_budget(rs, h.summary.shape, settings.max_window)
    arr = h.read_window(rs)
    return {
        "id": dataset_id,
        "slice": slice,
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
    if isinstance(adapter, _ArrowSpaceAdapter) and dataset_id in adapter._cache:
        stats = adapter.stats_data(dataset_id)
        return {"id": dataset_id, "backend": adapter.backend, "stats": stats}
    try:
        data = adapter.sidecar_stats(h.dataset_path)
        return {"id": dataset_id, "backend": "sidecar", "stats": data}
    except Exception:
        s = h.summary
        return {
            "id": dataset_id,
            "backend": "none",
            "stats": {"shape": list(s.shape), "dtype": s.dtype},
        }


@router.get("/datasets/{dataset_id:path}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if isinstance(adapter, _ArrowSpaceAdapter) and dataset_id in adapter._cache:
        data = adapter.manifold_data(dataset_id)
        return {"id": dataset_id, "backend": adapter.backend, "manifold": data}
    try:
        data = adapter.sidecar_manifold(h.dataset_path)
        return {"id": dataset_id, "backend": "sidecar", "manifold": data}
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No manifold data available for dataset '{dataset_id}'.",
        )


@router.get("/datasets/{dataset_id:path}/search")
def dataset_search_sidecar(
    dataset_id: str,
    q: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    results = adapter.sidecar_search(h.dataset_path, q, limit=limit)
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
    """Build (or rebuild) the ArrowSpace graph-Laplacian index.

    FIX: catch ValueError from adapter (e.g. 1-D array) and return 422.
    FIX: response returns graph_params flat alongside nitems/nfeatures/nclusters.
    """
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
    return {"id": dataset_id, "backend": adapter.backend, **data}


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


@router.post("/datasets/{dataset_id:path}/search")
def dataset_search(
    dataset_id: str,
    body: SearchRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search(dataset_id, body.model_dump())
    return {"id": dataset_id, **data}


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
    data = adapter.search_batch(dataset_id, body.model_dump())
    return {"id": dataset_id, **data}


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

    Use this endpoint to force a clean rebuild after the underlying Zarr
    array has changed, or to free disk space.

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
