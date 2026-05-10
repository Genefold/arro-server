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
GET  /api/prompts/audit                      -- full audit payload: degree stats, Fiedler, PCA 2D
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
    engine    = PromptSearchEngine.get()
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
        "nitems":         engine.aspace.nitems,
        "nclusters":      engine.aspace.nclusters,
        "lambdas":        lam,
        "lambdas_sorted": lam_sort,
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


@router.get("/prompts/audit")
def prompt_audit(
    pca_sample: int = Query(2000, ge=100, le=20000,
        description="Max number of nodes to include in the PCA 2D scatter. "
                    "Full 20k is ~4 MB JSON; 2000 is enough for the overview chart."),
) -> dict[str, Any]:
    """Full Audit Layer payload — mirrors the dataset inspection cells in the notebook.

    Heavy computation runs once; results are NOT cached across requests (the
    engine singleton already holds aspace+gl in RAM, so this is CPU-only).

    Returns
    -------
    index_stats       : nitems, nfeatures, nclusters, graph_params
    lambdas_stats     : min, max, mean, median, p25, p75, p95, full array
    degree_stats      : min, max, mean, std, cv, p10, p25, p50, p75, p90,
                        hub_count, hub_fraction, tail_count, tail_fraction
    fiedler           : fiedler_value, spectral_gap, connectivity_label
    gl_csr            : nnz, n_edges, sparsity
    scatter_pca       : list of {x, y, lambda, degree, id} for the UI scatter plot
                        (sampled to pca_sample points, stratified by lambda percentile)
    pca_variance      : [pc1_variance_ratio, pc2_variance_ratio]
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from sklearn.decomposition import PCA

    engine = PromptSearchEngine.get()
    lam    = np.array(engine.aspace.lambdas(), dtype=np.float64)

    # ── GL → scipy CSR ──────────────────────────────────────────────────────
    raw = engine.gl.to_csr()
    try:
        gl_shape_tuple = engine.gl.shape
    except TypeError:
        n = engine.gl.nnodes
        gl_shape_tuple = (n, n)
    L = sp.csr_matrix(
        (np.asarray(raw[0]), np.asarray(raw[1]), np.asarray(raw[2])),
        shape=gl_shape_tuple,
    )
    n_nodes  = L.shape[0]
    nnz      = int(L.nnz)
    n_edges  = (nnz - n_nodes) // 2
    sparsity = 1.0 - nnz / (n_nodes * n_nodes)
    degrees  = np.array(L.diagonal(), dtype=np.float64)

    # ── lambda stats ────────────────────────────────────────────────────────
    lp = lambda p: float(np.percentile(lam, p))
    lambdas_stats = {
        "min":    float(lam.min()),
        "max":    float(lam.max()),
        "mean":   float(lam.mean()),
        "median": float(np.median(lam)),
        "p25":    lp(25), "p60": lp(60), "p75": lp(75), "p95": lp(95),
        "values": lam.tolist(),          # full 20k array for histogram
    }

    # ── degree stats ────────────────────────────────────────────────────────
    dp = lambda p: float(np.percentile(degrees, p))
    p10d, p90d = dp(10), dp(90)
    hub_count  = int((degrees > p90d).sum())
    tail_count = int((degrees < p10d).sum())
    degree_stats = {
        "min":           float(degrees.min()),
        "max":           float(degrees.max()),
        "mean":          float(degrees.mean()),
        "std":           float(degrees.std()),
        "cv":            float(degrees.std() / (degrees.mean() + 1e-9)),
        "p10":           p10d, "p25": dp(25), "p50": dp(50),
        "p75":           dp(75), "p90": p90d,
        "hub_count":     hub_count,
        "hub_fraction":  round(hub_count  / n_nodes * 100, 2),
        "tail_count":    tail_count,
        "tail_fraction": round(tail_count / n_nodes * 100, 2),
    }

    # ── Fiedler (algebraic connectivity) ──────────────────────────────────
    try:
        diag_d     = np.where(degrees > 1e-12, degrees, 1e-12)
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_d))
        L_norm     = d_inv_sqrt @ L @ d_inv_sqrt
        eigs       = spla.eigsh(L_norm, k=6, which="SM",
                                return_eigenvectors=False, tol=1e-5, maxiter=3000)
        eigs_s     = sorted(np.real(eigs))
        fiedler_v  = max(0.0, eigs_s[1])
        spec_gap   = eigs_s[2] - eigs_s[1] if len(eigs_s) > 2 else 0.0
    except Exception:
        fiedler_v, spec_gap = 0.0, 0.0

    def _fiedler_label(f: float) -> str:
        if f < 0.01:  return "Near-disconnected (multiple isolated clusters)"
        if f < 0.05:  return "Weakly connected (eps may be too tight)"
        if f < 0.20:  return "Moderately connected"
        return "Well connected"

    fiedler = {
        "value":             round(fiedler_v, 6),
        "spectral_gap":      round(spec_gap,  6),
        "connectivity_label": _fiedler_label(fiedler_v),
    }

    # ── PCA 2D scatter (sampled, stratified by lambda percentile) ──────────
    sample_n = min(pca_sample, n_nodes)
    # stratified sampling: split into 4 lambda quantile buckets, sample evenly
    quartiles  = np.percentile(lam, [25, 50, 75])
    buckets    = np.digitize(lam, quartiles)          # 0,1,2,3
    per_bucket = max(1, sample_n // 4)
    idx_parts  = []
    for b in range(4):
        pool = np.where(buckets == b)[0]
        if len(pool) > 0:
            chosen = np.random.default_rng(42).choice(
                pool, size=min(per_bucket, len(pool)), replace=False
            )
            idx_parts.append(chosen)
    sample_idx = np.concatenate(idx_parts)[:sample_n]

    embs_sample = engine.embs[sample_idx].astype(np.float32)
    pca         = PCA(n_components=2, random_state=42)
    xy          = pca.fit_transform(embs_sample)
    var         = pca.explained_variance_ratio_

    scatter = [
        {
            "x":      round(float(xy[i, 0]), 5),
            "y":      round(float(xy[i, 1]), 5),
            "lambda": round(float(lam[sample_idx[i]]), 6),
            "degree": round(float(degrees[sample_idx[i]]), 4),
            "id":     engine.ids[sample_idx[i]],
        }
        for i in range(len(sample_idx))
    ]

    return {
        "index_stats": {
            "nitems":    int(engine.aspace.nitems),
            "nfeatures": int(engine.aspace.nfeatures),
            "nclusters": int(engine.aspace.nclusters),
            "graph_params": engine.gl.graph_params,
        },
        "lambdas_stats":  lambdas_stats,
        "degree_stats":   degree_stats,
        "fiedler":        fiedler,
        "gl_csr": {
            "nnz":      nnz,
            "n_edges":  n_edges,
            "sparsity": round(sparsity, 8),
        },
        "scatter_pca":    scatter,
        "pca_variance":   [round(float(v), 4) for v in var],
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
