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

GET  /api/prompts/health                     -- embedder + engine readiness
GET  /api/prompts/warm                       -- build aspace+gl, return index stats
GET  /api/prompts/lambdas                    -- eigenvalue distribution for prompt corpus
GET  /api/prompts/graph_laplacian            -- GL metadata for prompt corpus
GET  /api/prompts/audit                      -- full audit payload: degree stats, Fiedler, PCA 2D
POST /api/prompts/search                     -- LEAF kaban semantic search (pre-embedded vector)
POST /api/prompts/nl_search                  -- LEAF kaban NL search (server embeds query)
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
    NLSearchRequest,
    PromptSearchRequest,
    PromptSearchResponse,
    PromptSearchResult,
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
    # Check prompt engine + embedder readiness without triggering cold-start
    engine_ready    = PromptSearchEngine._instance is not None
    embedder_ready  = False
    try:
        from ..embedder import EmbedderService
        embedder_ready = EmbedderService._instance is not None
    except Exception:
        pass

    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": load_arrowspace().backend,
        "arrowspace_available": load_arrowspace().available,
        "data_roots": list(settings.resolved_roots.keys()),
        "prompt_engine_ready": engine_ready,
        "embedder_ready": embedder_ready,
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
        "attrs": h.attrs,
    }


@router.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int  = Query(-1),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if limit < 0:
        limit = settings.default_window
    limit = min(limit, settings.max_window)
    arr   = h.read(offset=offset, limit=limit)
    return array_to_payload(arr, offset=offset)


@router.get("/datasets/{dataset_id}/slice")
def dataset_slice(
    dataset_id: str,
    spec: str            = Query(..., description="NumPy-style slice, e.g. '0:10,2:5'."),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    h   = reg.open(dataset_id)
    try:
        slices = parse_slice(spec, h.ndim)
    except InvalidSlice as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        slices = enforce_window_budget(
            slices, h.shape, h.ndim,
            max_rows=settings.max_window,
            trailing_product=trailing_product,
        )
    except DatasetNotSliceable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    arr = h.slice(slices)
    return array_to_payload(arr, offset=0)


@router.get("/datasets/{dataset_id}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.sidecar_stats(h.path)
    except FileNotFoundError:
        return {}


@router.get("/datasets/{dataset_id}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.sidecar_manifold(h.path)
    except FileNotFoundError:
        return {}


@router.get("/datasets/{dataset_id}/search")
def dataset_keyword_search(
    dataset_id: str,
    q: str               = Query(..., description="Keyword query."),
    limit: int           = Query(20, ge=1, le=200),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    results = adapter.sidecar_search(h.path, q, limit=limit)
    return {"query": q, "results": results}


@router.post("/datasets/{dataset_id}/index")
def build_index(
    dataset_id: str,
    body: IndexBuildRequest | None = None,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    h            = reg.open(dataset_id)
    graph_params = (body.graph_params if body else None) or DEFAULT_GRAPH_PARAMS
    store        = Path(settings.index_store) / dataset_id
    store.mkdir(parents=True, exist_ok=True)
    try:
        result = adapter.build_index(h, graph_params=graph_params, store=store)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Index building requires the arrowspace package. "
                   "The sidecar adapter is read-only.",
        ) from None
    return result


@router.get("/datasets/{dataset_id}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.lambdas(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/graph_laplacian")
def dataset_graph_laplacian(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.graph_laplacian_info(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/items")
def dataset_items(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.items(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/items/{item_index}")
def dataset_item(
    dataset_id: str,
    item_index: int,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.item(h.path, item_index)
    except (FileNotFoundError, IndexError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/datasets/{dataset_id}/search")
def spectral_search(
    dataset_id: str,
    body: SearchRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h  = reg.open(dataset_id)
    q  = np.asarray(body.vector, dtype=np.float64)
    try:
        results = adapter.search(h.path, q, tau=body.tau)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="spectral search requires arrowspace package") from None
    return {"results": results}


@router.post("/datasets/{dataset_id}/search/energy")
def energy_search(
    dataset_id: str,
    body: SearchEnergyRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h  = reg.open(dataset_id)
    q  = np.asarray(body.vector, dtype=np.float64)
    try:
        results = adapter.search_energy(h.path, q, k=body.k)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="energy search requires arrowspace package") from None
    return {"results": results}


@router.post("/datasets/{dataset_id}/search/hybrid")
def hybrid_search(
    dataset_id: str,
    body: SearchHybridRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h  = reg.open(dataset_id)
    q  = np.asarray(body.vector, dtype=np.float64)
    try:
        results = adapter.search_hybrid(h.path, q, alpha=body.alpha)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="hybrid search requires arrowspace package") from None
    return {"results": results}


@router.post("/datasets/{dataset_id}/search/linear")
def linear_search(
    dataset_id: str,
    body: SearchLinearRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h  = reg.open(dataset_id)
    q  = np.asarray(body.vector, dtype=np.float64)
    try:
        results = adapter.search_linear(h.path, q, k=body.k)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="linear search requires arrowspace package") from None
    return {"results": results}


@router.post("/datasets/{dataset_id}/search/batch")
def batch_search(
    dataset_id: str,
    body: SearchBatchRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h       = reg.open(dataset_id)
    vectors = [np.asarray(v, dtype=np.float64) for v in body.vectors]
    try:
        results = adapter.search_batch(h.path, vectors, tau=body.tau)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="batch search requires arrowspace package") from None
    return {"results": results}


# ---------------------------------------------------------------------------
# Spot endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/spot/motives/eigen")
def spot_motives_eigen(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_motives_eigen(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/motives/energy")
def spot_motives_energy(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_motives_energy(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/subgraphs/centroids")
def spot_subgraph_centroids(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_subgraph_centroids(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/subgraphs/motives")
def spot_subgraph_motives(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_subgraph_motives(h.path)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Prompt / LEAF Kaban endpoints
# ---------------------------------------------------------------------------


def _get_engine() -> PromptSearchEngine:
    """Return the warm engine or raise 503 with a helpful message."""
    try:
        return PromptSearchEngine.get()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Prompt search engine is not ready: {exc}. "
                "Set ARRO_SERVER_PROMPT_DATA_DIR to the data directory and restart."
            ),
        ) from exc


def _get_embedder():
    """Return the warm embedder or raise 503 with a helpful message."""
    try:
        from ..embedder import EmbedderService
        return EmbedderService.get()
    except (ImportError, Exception) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embedder service is not ready: {exc}. "
                "Ensure sentence-transformers is installed: pip install 'arro-server[nlp]'"
            ),
        ) from exc


@router.get("/prompts/health")
def prompts_health() -> dict[str, Any]:
    """Return readiness status for the embedder and prompt search engine.

    Does NOT trigger a cold-start — use GET /api/prompts/warm for that.
    """
    engine_ready   = PromptSearchEngine._instance is not None
    embedder_ready = False
    embedder_model = None
    try:
        from ..embedder import EmbedderService
        embedder_ready = EmbedderService._instance is not None
        if embedder_ready:
            embedder_model = EmbedderService._instance.model_name  # type: ignore[union-attr]
    except Exception:
        pass

    status = "ready" if (engine_ready and embedder_ready) else "warming"
    return {
        "status": status,
        "prompt_engine_ready": engine_ready,
        "embedder_ready": embedder_ready,
        "embedder_model": embedder_model,
    }


@router.get("/prompts/warm")
def prompts_warm() -> dict[str, Any]:
    """Trigger (or confirm) warm-up of the PromptSearchEngine and EmbedderService.

    Call this once after deploy to front-load the index-build and model-load
    cost before real user traffic arrives.  Safe to call repeatedly.
    """
    engine    = _get_engine()
    _embedder = _get_embedder()
    return {
        "status": "warm",
        "nitems":    engine.aspace.nitems,
        "nfeatures": engine.aspace.nfeatures,
        "nclusters": engine.aspace.nclusters,
    }


@router.get("/prompts/lambdas")
def prompts_lambdas() -> dict[str, Any]:
    """Eigenvalue distribution of the prompt corpus graph Laplacian."""
    engine = _get_engine()
    try:
        lambdas = engine.gl.lambdas().tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute lambdas: {exc}") from exc
    return {"lambdas": lambdas, "n": len(lambdas)}


@router.get("/prompts/graph_laplacian")
def prompts_graph_laplacian() -> dict[str, Any]:
    """Metadata about the prompt corpus graph Laplacian."""
    engine = _get_engine()
    nnodes = int(engine.gl.nnodes)
    try:
        gl_shape = list(engine.gl.shape)
    except TypeError:
        gl_shape = [nnodes, nnodes]
    return {
        "nitems":    engine.aspace.nitems,
        "nfeatures": engine.aspace.nfeatures,
        "nclusters": engine.aspace.nclusters,
        "gl_nodes":  nnodes,
        "gl_shape":  gl_shape,
    }


@router.get("/prompts/audit")
def prompts_audit() -> dict[str, Any]:
    """Full audit payload: degree stats, Fiedler vector, PCA 2D scatter.

    Mirrors the cells in search_engine_demo.ipynb that visualise the corpus.
    """
    engine = _get_engine()
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="scikit-learn is required for /prompts/audit.",
        ) from exc

    # Degree stats
    try:
        gl_dense = engine.gl.to_dense()
        degrees  = np.array(gl_dense.sum(axis=1)).ravel() + gl_dense.diagonal()
    except Exception:
        degrees = np.zeros(engine.aspace.nitems)

    # Fiedler vector (second-smallest eigenvector)
    try:
        lambdas = engine.gl.lambdas()
        fiedler_val = float(sorted(lambdas)[1]) if len(lambdas) > 1 else 0.0
    except Exception:
        fiedler_val = 0.0

    # PCA 2D projection
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(engine.embs).tolist()

    return {
        "degree_stats": {
            "min":  float(degrees.min()),
            "max":  float(degrees.max()),
            "mean": float(degrees.mean()),
            "std":  float(degrees.std()),
        },
        "fiedler_value": fiedler_val,
        "pca_2d": coords,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "ids": engine.ids,
    }


@router.post("/prompts/search", response_model=PromptSearchResponse)
def prompts_search(body: PromptSearchRequest) -> PromptSearchResponse:
    """Semantic search with a pre-embedded 768-d nomic vector.

    For NL text queries use POST /api/prompts/nl_search instead.
    """
    engine = _get_engine()
    q      = np.asarray(body.vector, dtype=np.float64)
    try:
        raw = engine.search(q, k=body.k, tau=body.tau, alpha=body.alpha, lam=body.lam)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PromptSearchResponse(
        query=None,
        k=body.k,
        tau=body.tau,
        lam=body.lam,
        results=[PromptSearchResult(**r) for r in raw],
        result_count=len(raw),
    )


@router.post("/prompts/nl_search", response_model=PromptSearchResponse)
def prompts_nl_search(body: NLSearchRequest) -> PromptSearchResponse:
    """Natural-language semantic search.

    The server embeds `query` using EmbedderService
    (nomic-ai/nomic-embed-text-v1.5, SDPA) and runs the ArrowSpace spectral
    search + MMR rerank pipeline.  This is the primary endpoint for all
    frontend consumers.

    Request body example::

        {"query": "how to write a DALL-E prompt for minimalist art", "k": 10}

    Response envelope: PromptSearchResponse with result_count results,
    each carrying _score, _salience, _tau metadata.
    """
    embedder = _get_embedder()
    engine   = _get_engine()

    query_vec = embedder.embed(body.query)
    try:
        raw = engine.search(
            query_vec,
            k=body.k,
            tau=body.tau,
            alpha=body.alpha,
            lam=body.lam,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PromptSearchResponse(
        query=body.query,
        k=body.k,
        tau=body.tau,
        lam=body.lam,
        results=[PromptSearchResult(**r) for r in raw],
        result_count=len(raw),
    )
