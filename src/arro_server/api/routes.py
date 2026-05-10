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

GET  /api/prompts/warm                       -- build aspace+gl, return index stats
GET  /api/prompts/lambdas                    -- eigenvalue distribution for prompt corpus
GET  /api/prompts/graph_laplacian            -- GL metadata for prompt corpus
POST /api/prompts/search                     -- LEAF kaban semantic search
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import __version__
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS, ArrowSpaceAdapter, _ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotSliceable, InvalidSlice
from ..search_engine import PromptSearchEngine
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice, trailing_product
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
from .schemas import (
    IndexBuildRequest,
    PromptSearchRequest,
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
    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": load_arrowspace().backend,
        "arrowspace_available": load_arrowspace().available,
        "data_roots": list(settings.resolved_roots.keys()),
    }


# ---------------------------------------------------------------------------
# Dataset discovery + raw Zarr access
# ---------------------------------------------------------------------------


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


@router.get("/datasets/{dataset_id}/metadata")
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


@router.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if not h.summary.shape:
        raise DatasetNotSliceable(dataset_id, "dataset has no shape")
    eff_limit = limit or settings.default_window
    rs = parse_slice(None, h.summary.shape, offset=offset, limit=eff_limit)
    try:
        enforce_window_budget(rs, settings.max_window * max(1, trailing_product(h.summary.shape)))
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


@router.get("/datasets/{dataset_id}/slice")
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
    except ValueError as e:
        raise InvalidSlice(str(e)) from e
    arr = h.read_window(rs)
    return {"id": h.summary.dataset_id, "slice": spec, "out_shape": list(arr.shape), "data": array_to_payload(arr)}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    base_stats = h.stats()
    arrowspace_stats: dict[str, Any] = {}
    if isinstance(adapter, _ArrowSpaceAdapter):
        try:
            arrowspace_stats = adapter.stats_data(dataset_id)
        except Exception:
            pass
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        "stats": {**base_stats, **arrowspace_stats},
    }


# ---------------------------------------------------------------------------
# Manifold (sidecar or live)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    dataset_path = h.fs_path  # type: ignore[attr-defined]
    live_data: dict[str, Any] | None = None
    if isinstance(adapter, _ArrowSpaceAdapter):
        try:
            live_data = adapter.manifold_data(dataset_id)
        except Exception:
            pass
    if live_data is not None:
        manifold_payload, source = live_data, "live"
    else:
        try:
            manifold_payload = adapter.sidecar_manifold(dataset_path)
            source = "sidecar"
        except Exception as e:
            manifold_payload = {"unavailable": str(e)}
            source = "unavailable"
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        "source": source,
        "manifold": manifold_payload,
    }


# ---------------------------------------------------------------------------
# Sidecar keyword search (GET)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/search")
def dataset_search_sidecar(
    dataset_id: str,
    q: str = Query(...),
    limit: int = Query(20, ge=1, le=1000),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    dataset_path = h.fs_path  # type: ignore[attr-defined]
    results = adapter.sidecar_search(dataset_path, q, limit=limit)
    return {"id": dataset_id, "q": q, "results": results}


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/index")
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


# ---------------------------------------------------------------------------
# Lambdas
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.lambdas(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


# ---------------------------------------------------------------------------
# Graph Laplacian info
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/graph_laplacian")
def dataset_graph_laplacian(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.graph_laplacian_info(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


# ---------------------------------------------------------------------------
# Item retrieval
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/items")
def dataset_get_all_items(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.get_all_items(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id}/items/{item_index}")
def dataset_get_item(
    dataset_id: str,
    item_index: int,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.get_item(dataset_id, item_index)
    return {"id": dataset_id, "backend": adapter.backend, **data}


# ---------------------------------------------------------------------------
# Vector search variants
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/search")
def dataset_search_vector(
    dataset_id: str,
    body: SearchRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search(dataset_id, body.model_dump())
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


@router.post("/datasets/{dataset_id}/search/energy")
def dataset_search_energy(
    dataset_id: str,
    body: SearchEnergyRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_energy(dataset_id, body.model_dump())
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


@router.post("/datasets/{dataset_id}/search/hybrid")
def dataset_search_hybrid(
    dataset_id: str,
    body: SearchHybridRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_hybrid(dataset_id, body.model_dump())
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


@router.post("/datasets/{dataset_id}/search/linear")
def dataset_search_linear(
    dataset_id: str,
    body: SearchLinearRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_linear_sorted(dataset_id, body.model_dump())
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


@router.post("/datasets/{dataset_id}/search/batch")
def dataset_search_batch(
    dataset_id: str,
    body: SearchBatchRequest,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.search_batch(dataset_id, body.model_dump())
    return {"id": dataset_id, "backend": adapter.backend, "arrowspace_available": adapter.available, **data}


# ---------------------------------------------------------------------------
# Spot methods
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/spot/motives/eigen")
def dataset_spot_motives_eigen(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_motives_eigen(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id}/spot/motives/energy")
def dataset_spot_motives_energy(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_motives_energy(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id}/spot/subgraphs/centroids")
def dataset_spot_subg_centroids(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_subg_centroids(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


@router.get("/datasets/{dataset_id}/spot/subgraphs/motives")
def dataset_spot_subg_motives(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    data = adapter.spot_subg_motives(dataset_id)
    return {"id": dataset_id, "backend": adapter.backend, **data}


# ---------------------------------------------------------------------------
# LEAF kaban — Prompt corpus index + inspection + search
# ---------------------------------------------------------------------------


@router.get("/prompts/warm")
def prompt_warm() -> dict[str, Any]:
    """Initialise (or confirm already-initialised) PromptSearchEngine.

    Loads embeddings, builds the ArrowSpace graph-Laplacian index from the
    tuner-optimised eps & k, and returns index statistics.
    Safe to call repeatedly — the singleton is only built once per process.
    """
    engine = PromptSearchEngine.get()
    nclusters = int(engine.aspace.nclusters)
    nitems    = int(engine.aspace.nitems)
    nfeatures = int(engine.aspace.nfeatures)
    nnodes    = int(engine.gl.nnodes)
    try:
        gl_shape = list(engine.gl.shape)
    except TypeError:
        gl_shape = [nnodes, nnodes]
    return {
        "status":    "ready",
        "nitems":    nitems,
        "nfeatures": nfeatures,
        "nclusters": nclusters,
        "gl_nnodes": nnodes,
        "gl_shape":  gl_shape,
    }


@router.get("/prompts/lambdas")
def prompt_lambdas() -> dict[str, Any]:
    """Return the full eigenvalue spectrum of the prompt corpus graph-Laplacian.

    Triggers a warm-up build if the engine has not been initialised yet.
    Useful for inspecting spectral structure and validating index quality.
    """
    engine   = PromptSearchEngine.get()
    lam      = [float(v) for v in engine.aspace.lambdas()]
    lam_sort = [[float(v), int(i)] for v, i in engine.aspace.lambdas_sorted()]
    return {
        "nitems":          engine.aspace.nitems,
        "nclusters":       engine.aspace.nclusters,
        "lambdas":         lam,
        "lambdas_sorted":  lam_sort,
    }


@router.get("/prompts/graph_laplacian")
def prompt_graph_laplacian() -> dict[str, Any]:
    """Return graph-Laplacian metadata for the prompt corpus index.

    Triggers a warm-up build if the engine has not been initialised yet.
    """
    engine = PromptSearchEngine.get()
    nnodes = int(engine.gl.nnodes)
    try:
        gl_shape = list(engine.gl.shape)
    except TypeError:
        gl_shape = [nnodes, nnodes]
    return {
        "nnodes":       nnodes,
        "shape":        gl_shape,
        "graph_params": engine.gl.graph_params,
    }


@router.post("/prompts/search")
def prompt_search(body: PromptSearchRequest) -> dict:
    """Semantic search over the 20k prompt corpus.

    Input:  768-dim nomic-embed-text-v1.5 vector (embedded by caller).
    Output: top-k prompt JSON records enriched with _score, _salience, _tau.

    Graph topology (eps, k) is fixed at startup from the latest tuner run.
    tau controls spectral sharpness at query time (default 0.75, range 0–5).
    """
    engine  = PromptSearchEngine.get()
    results = engine.search(
        query_vec=np.array(body.vector, dtype=np.float64),
        k=body.k,
        tau=body.tau,
        alpha=body.alpha,
    )
    return {"count": len(results), "results": results}
