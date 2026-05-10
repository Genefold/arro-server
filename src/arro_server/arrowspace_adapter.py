"""Adapter for arrowspace / ArrowSpace graph-Laplacian index.

The ``arrowspace`` package (pip install arrowspace,
repo: https://github.com/tuned-org-uk/pyarrowspace) may not be available in
every environment.  We never import it at module load — :func:`load` attempts
the import lazily and returns a stub adapter that falls back to the sidecar
JSON adapter.

Real arrowspace API (confirmed from package introspection):

    from arrowspace import ArrowSpaceBuilder
    aspace, gl = ArrowSpaceBuilder().build(graph_params, np_array_float64)

ArrowSpace object public surface::

    aspace.nitems          int
    aspace.nfeatures       int
    aspace.nclusters       int
    aspace.lambdas()       -> np.ndarray          eigenvalue vector
    aspace.lambdas_sorted()-> List[(float, int)]  sorted (value, original_index)
    aspace.search(vec: np.ndarray[float64], gl, tau: float)
                           -> List[(int, float)]  (index, score)
    aspace.search_batch / search_hybrid / search_energy / search_linear_sorted
    aspace.get_item(i)     -> item at position i
    aspace.get_all_items() -> all items
    aspace.spot_motives_eigen / spot_motives_energy / spot_subg_motives / spot_subg_centroids

GraphLaplacian object public surface::

    gl.nnodes              int
    gl.shape               (rows, cols)
    gl.graph_params        dict
    gl.to_csr()            -> (data, indices, indptr, shape)  numpy arrays
    gl.to_dense()          -> np.ndarray  2D float32

The raw dataset stays as the Zarr v3 array (served by the existing Zarr
backend).  The *graph Laplacian* produced by arrowspace is persisted as a
second Zarr v3 array (CSR components) under::

    <ARRO_SERVER_INDEX_STORE>/<dataset_id_slug>/
        data.zarr    # gl.to_csr() -> data array  (float32)
        indices.zarr # CSR column indices         (int64)
        indptr.zarr  # CSR row pointers           (int64)
        meta.json    # {nitems, nfeatures, nclusters, shape}

The ``ArrowSpace`` object itself is kept in-process memory (bounded by
``ARRO_SERVER_INDEX_CACHE_SIZE``, default 8).  The oldest entry is evicted
when the limit is reached.  On the next server start the Zarr arrays can be
loaded back (Phase 2 work).

In-memory cache
---------------
The live adapter uses a simple ``OrderedDict``-backed LRU bounded by
``Settings.index_cache_size``.  Eviction is logged at INFO level so operators
can tune the limit without surprises.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MetadataUnavailable, OptionalDependencyMissing

log = logging.getLogger(__name__)

# Default graph params passed to ArrowSpaceBuilder.build()
DEFAULT_GRAPH_PARAMS: dict[str, Any] = {
    "eps": 1.0,
    "k": 6,
    "topk": 3,
    "p": 2.0,
    "sigma": 1.0,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ArrowSpaceAdapter(ABC):
    """Common interface for all ArrowSpace backend implementations."""

    def __init__(self, *, available: bool, backend: str) -> None:
        self.available = available
        self.backend = backend

    @abstractmethod
    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def lambdas(self, dataset_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]: ...

    @abstractmethod
    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]: ...

    @abstractmethod
    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Sidecar JSON adapter (no package dependency)
# ---------------------------------------------------------------------------


class _SidecarAdapter(ArrowSpaceAdapter):
    """Reads pre-written JSON sidecar files from ``<dataset>/_arrowspace/``.

    backend = "sidecar"
    available = True  (sidecar files are always readable when present)
    """

    def __init__(self) -> None:
        super().__init__(available=True, backend="sidecar")

    @staticmethod
    def _read(dataset_path: Path, filename: str) -> dict[str, Any]:
        sidecar = dataset_path / "_arrowspace" / filename
        if not sidecar.exists():
            raise MetadataUnavailable(f"{sidecar} not found")
        return json.loads(sidecar.read_text())

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "manifold.json")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "stats.json")

    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Keyword search against ``_arrowspace/index.json``.

        Raises :class:`~.errors.MetadataUnavailable` (404) when the sidecar
        index file is absent.  Matching is case-insensitive substring search
        against each item's ``id`` and ``tags`` fields.
        """
        data = self._read(dataset_path, "index.json")
        items: list[dict[str, Any]] = data.get("items", [])
        q_lower = q.lower()
        results = []
        for item in items:
            item_id: str = str(item.get("id", ""))
            tags: list[str] = [str(t) for t in item.get("tags", [])]
            if q_lower in item_id.lower() or any(q_lower in t.lower() for t in tags):
                results.append({"id": item_id, "tags": tags})
            if len(results) >= limit:
                break
        return results

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise OptionalDependencyMissing(
            "arrowspace",
            "build_index (install arrowspace package: pip install arrowspace)",
        )

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "lambdas")

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing(
            "arrowspace",
            "vector search (install arrowspace or use GET /search with sidecar index.json)",
        )


# ---------------------------------------------------------------------------
# No-op adapter (arrowspace not installed, no sidecar)
# ---------------------------------------------------------------------------


class _UnavailableAdapter(ArrowSpaceAdapter):
    """backend = "none", available = False."""

    def __init__(self) -> None:
        super().__init__(available=False, backend="none")

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "build_index")

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "lambdas")

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search")

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "manifold sidecar")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "stats sidecar")

    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        raise OptionalDependencyMissing("arrowspace", "sidecar search")


# ---------------------------------------------------------------------------
# Bounded LRU cache helper
# ---------------------------------------------------------------------------


@dataclass
class _IndexEntry:
    """In-memory cache slot for one built index."""

    aspace: Any  # arrowspace.ArrowSpace
    gl: Any      # arrowspace.GraphLaplacian
    nitems: int
    nfeatures: int
    nclusters: int


class _LRUIndexCache:
    """Simple OrderedDict-backed LRU cache with a configurable max size."""

    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = max(1, maxsize)
        self._data: OrderedDict[str, _IndexEntry] = OrderedDict()

    def get(self, key: str) -> _IndexEntry | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, entry: _IndexEntry) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = entry
        while len(self._data) > self._maxsize:
            evicted, _ = self._data.popitem(last=False)
            log.info("ArrowSpace index cache evicted '%s' (cache_size=%d)", evicted, self._maxsize)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Live arrowspace adapter
# ---------------------------------------------------------------------------


class _ArrowSpaceAdapter(ArrowSpaceAdapter):
    """Live adapter backed by the ``arrowspace`` package.

    backend = "arrowspace"
    available = True

    Real API (confirmed via package introspection):

        aspace, gl = ArrowSpaceBuilder().build(graph_params_dict, np_array_float64)

    ArrowSpace attributes/methods used here:
        .nitems, .nfeatures, .nclusters          — int scalars
        .lambdas()                               — np.ndarray of eigenvalues
        .lambdas_sorted()                        — List[(float, int)]
        .search(vec: np.ndarray, gl, tau: float) — List[(int, float)]

    GraphLaplacian attributes/methods used here:
        .nnodes, .shape                          — int / (int, int)
        .to_csr()                                — (data, indices, indptr, shape)
        .to_dense()                              — np.ndarray

    The adapter:
    1. Accepts a ``numpy.ndarray`` (already read from Zarr by the route handler).
    2. Calls ``ArrowSpaceBuilder().build(graph_params, array)``.
    3. Persists the graph-Laplacian CSR components as Zarr v3 arrays under
       ``<index_store>/<slug>/`` (best-effort; failures are logged only).
    4. Caches (ArrowSpace, GraphLaplacian) in an LRU bounded by
       ``Settings.index_cache_size`` (default 8).
    """

    def __init__(self, module: Any, cache_size: int = 8) -> None:
        super().__init__(available=True, backend="arrowspace")
        self._mod = module
        self._cache = _LRUIndexCache(maxsize=cache_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slug(dataset_id: str) -> str:
        """File-system-safe slug for a dataset_id."""
        return dataset_id.replace("/", "__").replace("\\", "__")

    def _persist_csr(
        self,
        index_store: Path,
        slug: str,
        gl: Any,
        meta: dict[str, Any],
    ) -> None:
        """Write CSR arrays + meta.json to ``index_store/<slug>/``.

        Failures here are non-fatal: the in-memory index is already cached.
        A warning is logged so operators can tune the limit without surprises.

        gl.to_csr() returns (data, indices, indptr, shape).
        """
        try:
            import zarr  # type: ignore
        except ImportError:
            log.warning("zarr not installed; graph-Laplacian will not be persisted")
            return

        try:
            csr_data, csr_indices, csr_indptr, csr_shape = gl.to_csr()

            dest = index_store / slug
            dest.mkdir(parents=True, exist_ok=True)

            for arr_name, arr_val in (
                ("data", np.asarray(csr_data, dtype=np.float32)),
                ("indices", np.asarray(csr_indices, dtype=np.int64)),
                ("indptr", np.asarray(csr_indptr, dtype=np.int64)),
            ):
                zarr_path = dest / f"{arr_name}.zarr"
                z = zarr.open(
                    str(zarr_path),
                    mode="w",
                    shape=arr_val.shape,
                    dtype=arr_val.dtype,
                    chunks=True,
                    zarr_format=3,
                )
                z[:] = arr_val

            meta_dict = dict(meta)
            meta_dict["csr_shape"] = list(csr_shape)
            (dest / "meta.json").write_text(json.dumps(meta_dict))
            log.info("Persisted graph-Laplacian CSR to %s", dest)
        except Exception:
            log.warning(
                "Failed to persist graph-Laplacian CSR for '%s'; "
                "in-memory index is still available for this server lifetime.",
                slug,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ArrowSpace graph-Laplacian index for *array*.

        The input *array* must be a 2-D float64 ndarray (rows = items,
        columns = features).  It is read from the Zarr backend by the
        route handler before calling this method.

        Returns dict with keys: nitems, nfeatures, nclusters.
        """
        gp = graph_params or DEFAULT_GRAPH_PARAMS
        arr64 = np.asarray(array, dtype=np.float64)
        if arr64.ndim != 2:  # noqa: PLR2004
            raise ValueError(
                f"arrowspace requires a 2-D array (items x features); got shape {arr64.shape}"
            )

        log.info(
            "Building arrowspace index for '%s' (shape=%s, params=%s)",
            dataset_id,
            arr64.shape,
            gp,
        )
        aspace, gl = self._mod.ArrowSpaceBuilder().build(gp, arr64)

        entry = _IndexEntry(
            aspace=aspace,
            gl=gl,
            nitems=int(aspace.nitems),
            nfeatures=int(aspace.nfeatures),
            nclusters=int(aspace.nclusters),
        )
        self._cache.put(dataset_id, entry)

        meta = {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
        }
        # Persistence is best-effort: failures are logged but do not bubble up.
        self._persist_csr(index_store, self._slug(dataset_id), gl, meta)

        return meta

    # ------------------------------------------------------------------
    # Cache retrieval helper
    # ------------------------------------------------------------------

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        """Retrieve a cached index entry or raise 404."""
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No index built for '{dataset_id}'. "
                "Call POST /api/datasets/{id}/index first."
            )
        return entry

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        """Return Laplacian eigenvalue distribution for a built index.

        Returns::

            {
                "nitems":        int,
                "lambdas":       List[float],        # all eigenvalues, original order
                "lambdas_sorted": [[float, int], ...]  # (value, original_index) sorted desc
            }
        """
        entry = self._get_entry(dataset_id)
        lam = list(entry.aspace.lambdas())
        lam_sorted = [[float(v), int(i)] for v, i in entry.aspace.lambdas_sorted()]
        return {
            "nitems": entry.nitems,
            "lambdas": [float(v) for v in lam],
            "lambdas_sorted": lam_sorted,
        }

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        """Vector search against the in-memory arrowspace index.

        *query* must contain ``vector`` (list of float64 values).
        Optional ``tau`` (float, default 1.0).

        Returns::

            {
                "backend": "arrowspace",
                "results": [{"index": int, "score": float}, ...]
            }
        """
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable(
                "arrowspace search requires 'vector' key (list of float64 values)"
            )
        tau = float(query.get("tau", 1.0))
        q_arr = np.asarray(vec, dtype=np.float64)
        hits = entry.aspace.search(q_arr, entry.gl, tau)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    def manifold_data(self, dataset_id: str) -> dict[str, Any]:
        """Return manifold summary from the in-memory index.

        Maps to ArrowSpace object state::

            nitems, nfeatures, nclusters  — scalar attributes
            lambdas_sorted()[:50]         — top-50 eigenvalues for topology overview
        """
        entry = self._get_entry(dataset_id)
        lam_sorted = [[float(v), int(i)] for v, i in entry.aspace.lambdas_sorted()]
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "lambdas_sorted": lam_sorted[:50],
        }

    def stats_data(self, dataset_id: str) -> dict[str, Any]:
        """Return graph-Laplacian statistics from the in-memory index.

        Maps to GraphLaplacian object state::

            gl.nnodes   — number of graph nodes
            gl.shape    — (rows, cols) of the Laplacian matrix
            aspace.*    — dataset dimension metadata
        """
        entry = self._get_entry(dataset_id)
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "gl_nodes": int(entry.gl.nnodes),
            "gl_shape": list(entry.gl.shape),
        }

    # ------------------------------------------------------------------
    # Sidecar helpers (delegate to the static sidecar reader)
    # ------------------------------------------------------------------

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "manifold.json")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "stats.json")

    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        return _SidecarAdapter().sidecar_search(dataset_path, q, limit=limit)


# ---------------------------------------------------------------------------
# Module-level factory + cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load() -> ArrowSpaceAdapter:
    """Return the best available ArrowSpace adapter.

    Priority order:
    1. ``arrowspace`` package if importable  -> :class:`_ArrowSpaceAdapter`
    2. Sidecar JSON files present            -> :class:`_SidecarAdapter`

    The result is cached for the process lifetime.
    Use ``reset_adapter_cache()`` in tests to reset between cases.
    """
    from .settings import get_settings

    try:
        import arrowspace as _mod  # type: ignore

        cache_size = get_settings().index_cache_size
        log.info("arrowspace package found; using live adapter (cache_size=%d)", cache_size)
        return _ArrowSpaceAdapter(_mod, cache_size=cache_size)
    except ImportError:
        log.info("arrowspace package not found; using sidecar adapter")
        return _SidecarAdapter()


def reset_adapter_cache() -> None:
    """Test / reload helper — clears the lru_cache on load()."""
    load.cache_clear()
