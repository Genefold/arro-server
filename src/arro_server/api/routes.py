"""API route handlers for arro-server.

All routes are mounted under the /api prefix (set on the router).
Dataset IDs use '--' as the root/path separator (e.g. 'main--matrix').

Endpoint map
------------
GET  /api/health                             -- liveness + dep status
GET  /api/datasets                           -- list all discovered datasets
GET  /api/datasets/{id}/metadata             -- shape, dtype, chunks, attrs
GET  /api/datasets/{id}/data                 -- row-window (offset/limit)
GET  /api/datasets/{id}/slice                -- numpy-style multi-axis slice
GET  /api/datasets/{id}/stats                -- basic array stats
GET  /api/datasets/{id}/manifold             -- ArrowSpace manifold (sidecar or live)
GET  /api/datasets/{id}/search               -- keyword search (sidecar only)
POST /api/datasets/{id}/index                -- build ArrowSpace graph-Laplacian index
GET  /api/datasets/{id}/lambdas              -- Laplacian eigenvalue distribution
POST /api/datasets/{id}/search               -- vector search against built index
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from .. import __version__
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS, ArrowSpaceAdapter, _ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotSliceable, InvalidSlice
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice, trailing_product
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
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
    """Liveness check.

    Returns service version, zarr availability, arrowspace backend name,
    and the list of configured data root labels.
    """
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
    """Row-oriented window over the leading axis. Suited to infinite scroll.

    ``limit`` and ``ARRO_SERVER_MAX_WINDOW`` are row counts (leading-axis
    elements).  For N-D arrays the total element budget is
    ``max_window * product(shape[1:])``.  Use ``/slice`` for precise
    multi-axis control.
    """
    h = reg.open(dataset_id)
    if not h.summary.shape:
        raise DatasetNotSliceable(dataset_id, "dataset has no shape (0-d or group)")
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
    spec: str = Query(..., alias="slice", description="Comma-separated per-axis slice spec"),
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
    return {
        "id": h.summary.dataset_id,
        "slice": spec,
        "out_shape": list(arr.shape),
        "data": array_to_payload(arr),
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Array statistics enriched with ArrowSpace graph-Laplacian stats when
    a live index has been built for this dataset.

    Always returns basic Zarr stats (shape, dtype, chunks, size).
    When a live arrowspace index is available, merges in nitems, nfeatures,
    nclusters, gl_nodes, gl_shape from the in-memory GraphLaplacian.

    Response includes:
        ``arrowspace_available`` -- bool, whether the arrowspace package is loaded
        ``backend``              -- "arrowspace" | "sidecar" | "none"
    """
    h = reg.open(dataset_id)
    base_stats = h.stats()

    # Attempt to enrich with live arrowspace stats if index exists
    arrowspace_stats: dict[str, Any] = {}
    if isinstance(adapter, _ArrowSpaceAdapter):
        try:
            arrowspace_stats = adapter.stats_data(dataset_id)
        except Exception:
            pass  # index not built yet — return basic stats only

    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        "stats": {**base_stats, **arrowspace_stats},
    }


# ---------------------------------------------------------------------------
# Sidecar manifold  (static JSON sidecar, no arrowspace package required)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """ArrowSpace manifold metadata for a dataset.

    Behaviour by backend:
    - ``arrowspace`` backend with a built index: returns live manifold data
      from the in-memory ArrowSpace object (nitems, nfeatures, nclusters,
      lambdas_sorted[:50]).
    - ``arrowspace`` backend without a built index, or ``sidecar`` backend:
      reads ``<dataset>/_arrowspace/manifold.json`` sidecar file.
    - ``none`` backend: returns {"unavailable": reason}.

    Response always includes:
        ``backend``              -- "arrowspace" | "sidecar" | "none"
        ``arrowspace_available`` -- bool
    """
    h = reg.open(dataset_id)
    dataset_path = h.fs_path  # type: ignore[attr-defined]

    # Try live manifold from in-memory index first (arrowspace backend only)
    live_data: dict[str, Any] | None = None
    if isinstance(adapter, _ArrowSpaceAdapter):
        try:
            live_data = adapter.manifold_data(dataset_id)
        except Exception:
            pass  # index not built yet — fall through to sidecar

    if live_data is not None:
        manifold_payload = live_data
        source = "live"
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
# Sidecar keyword search  (GET, query-string based, sidecar only)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/search")
def dataset_search_sidecar(
    dataset_id: str,
    q: str = Query(..., description="Keyword to match against sidecar index tags/ids."),
    limit: int = Query(20, ge=1, le=1000),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Keyword search against ``_arrowspace/index.json`` sidecar.

    This is a text-based substring match — it does NOT use the vector index.
    For vector similarity search, use ``POST /datasets/{id}/search``.

    Returns ``{id, q, results: [{id, tags}, ...]}``.
    Raises 404 when no sidecar index is present for the dataset.
    """
    h = reg.open(dataset_id)
    dataset_path = h.fs_path  # type: ignore[attr-defined]
    results = adapter.sidecar_search(dataset_path, q, limit=limit)
    return {
        "id": dataset_id,
        "q": q,
        "results": results,
    }


# ---------------------------------------------------------------------------
# ArrowSpace index lifecycle  (Task 1.3)
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/index")
def build_index(
    dataset_id: str,
    graph_params: dict[str, Any] | None = Body(
        default=None,
        examples=[DEFAULT_GRAPH_PARAMS],
        description=(
            "ArrowSpaceBuilder graph params. "
            "Omit to use server defaults: "
            "{eps: 1.0, k: 6, topk: 3, p: 2.0, sigma: 1.0}"
        ),
    ),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Build (or rebuild) the ArrowSpace graph-Laplacian index for a dataset.

    Reads the full Zarr array, calls ``ArrowSpaceBuilder().build()``, persists
    the graph-Laplacian CSR arrays under ``ARRO_SERVER_INDEX_STORE``, and
    caches the result in memory for fast /lambdas and /search access.

    The source Zarr array must be 2-D (rows = items, columns = features).

    Note: this endpoint is synchronous and reads the entire array into RAM.
    For very large arrays consider chunked ingestion (Phase 2 work).

    Returns::

        {
            "id": str,
            "built": true,
            "graph_params": {...},
            "nitems": int,
            "nfeatures": int,
            "nclusters": int
        }
    """
    h = reg.open(dataset_id)
    rs = parse_slice(None, h.summary.shape, offset=0, limit=h.summary.shape[0])
    arr = h.read_window(rs)

    index_store = Path(settings.index_store).expanduser().resolve()
    effective_params = graph_params or DEFAULT_GRAPH_PARAMS
    meta = adapter.build_index(
        dataset_id=dataset_id,
        array=arr,
        index_store=index_store,
        graph_params=effective_params,
    )
    return {
        "id": dataset_id,
        "built": True,
        "graph_params": effective_params,
        **meta,
    }


# ---------------------------------------------------------------------------
# ArrowSpace lambdas  (Task 1.5 — surface backend in response)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Return the Laplacian eigenvalue distribution for a built index.

    Requires a prior call to ``POST /datasets/{id}/index``.

    Returns::

        {
            "id": str,
            "backend": str,
            "arrowspace_available": bool,
            "nitems": int,
            "lambdas": [float, ...],
            "lambdas_sorted": [[float, int], ...]
        }
    """
    data = adapter.lambdas(dataset_id)
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        **data,
    }


# ---------------------------------------------------------------------------
# ArrowSpace vector search  (Task 1.4 — POST only for vector queries)
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/search")
def dataset_search_vector(
    dataset_id: str,
    body: dict[str, Any] = Body(
        ...,
        examples=[{"vector": [0.1, 0.2, 0.3], "tau": 1.0}],
        description=(
            "Search body: 'vector' (list[float]) required; "
            "'tau' (float, default 1.0) optional."
        ),
    ),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Vector similarity search against the in-memory ArrowSpace index.

    Requires a prior call to ``POST /datasets/{id}/index``.

    This is a POST endpoint because the query is a float64 vector, not a
    free-text string.  For text-based sidecar search use
    ``GET /datasets/{id}/search?q=...``.

    Body::

        {"vector": [f64, ...], "tau": 1.0}

    Returns::

        {
            "id": str,
            "backend": str,
            "arrowspace_available": bool,
            "results": [{"index": int, "score": float}, ...]
        }
    """
    data = adapter.search(dataset_id, body)
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "arrowspace_available": adapter.available,
        **data,
    }
