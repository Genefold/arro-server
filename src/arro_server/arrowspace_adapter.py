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
    aspace.search(vec, gl, tau)             -> List[(int, float)]
    aspace.search_batch(vecs, gl, tau)      -> List[List[(int, float)]]
    aspace.search_energy(vec, gl, k)        -> List[(int, float)]
    aspace.search_hybrid(vec, gl, alpha)    -> List[(int, float)]
    aspace.search_linear_sorted(vec, gl, k) -> List[(int, float)]
    aspace.get_item(i)     -> item at position i
    aspace.get_all_items() -> all items
    aspace.spot_motives_eigen()    -> List[(int, float)]
    aspace.spot_motives_energy()   -> List[(int, float)]
    aspace.spot_subg_centroids()   -> List[(int, float)]
    aspace.spot_subg_motives()     -> List[(int, float)]

GraphLaplacian object public surface::

    gl.nnodes              int
    gl.shape               (rows, cols)
    gl.graph_params        dict
    gl.to_csr()            -> (data, indices, indptr, shape)
    gl.to_dense()          -> np.ndarray  2D float32
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
from fastapi import HTTPException

from .errors import MetadataUnavailable, OptionalDependencyMissing

log = logging.getLogger(__name__)

DEFAULT_GRAPH_PARAMS: dict[str, Any] = {
    "eps": 1.0,
    "k": 6,
    "topk": 3,
    "p": 2.0,
    "sigma": 1.0,
}

DEFAULT_SEARCH_K: int = 10


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ArrowSpaceAdapter(ABC):
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
# Sidecar JSON adapter
# ---------------------------------------------------------------------------


class _SidecarAdapter(ArrowSpaceAdapter):
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

    def graph_laplacian_info(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "graph_laplacian_info")

    def get_item(self, dataset_id: str, idx: int) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "get_item")

    def get_all_items(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "get_all_items")

    def search_batch(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search_batch")

    def search_energy(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search_energy")

    def search_hybrid(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search_hybrid")

    def search_linear_sorted(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search_linear_sorted")

    def spot_motives_eigen(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "spot_motives_eigen")

    def spot_motives_energy(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "spot_motives_energy")

    def spot_subg_centroids(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "spot_subg_centroids")

    def spot_subg_motives(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "spot_subg_motives")


# ---------------------------------------------------------------------------
# No-op adapter
# ---------------------------------------------------------------------------


class _UnavailableAdapter(ArrowSpaceAdapter):
    def __init__(self) -> None:
        super().__init__(available=False, backend="none")

    def build_index(self, dataset_id, array, index_store, graph_params=None):
        raise OptionalDependencyMissing("arrowspace", "build_index")

    def lambdas(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "lambdas")

    def search(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search")

    def sidecar_manifold(self, dataset_path):
        raise OptionalDependencyMissing("arrowspace", "manifold sidecar")

    def sidecar_stats(self, dataset_path):
        raise OptionalDependencyMissing("arrowspace", "stats sidecar")

    def sidecar_search(self, dataset_path, q, *, limit=20):
        raise OptionalDependencyMissing("arrowspace", "sidecar search")

    def graph_laplacian_info(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "graph_laplacian_info")

    def get_item(self, dataset_id, idx):
        raise OptionalDependencyMissing("arrowspace", "get_item")

    def get_all_items(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "get_all_items")

    def search_batch(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search_batch")

    def search_energy(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search_energy")

    def search_hybrid(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search_hybrid")

    def search_linear_sorted(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search_linear_sorted")

    def spot_motives_eigen(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "spot_motives_eigen")

    def spot_motives_energy(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "spot_motives_energy")

    def spot_subg_centroids(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "spot_subg_centroids")

    def spot_subg_motives(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "spot_subg_motives")


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


@dataclass
class _IndexEntry:
    aspace: Any
    gl: Any
    nitems: int
    nfeatures: int
    nclusters: int


class _LRUIndexCache:
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
            log.info("ArrowSpace cache evicted '%s'", evicted)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Live adapter
# ---------------------------------------------------------------------------


class _ArrowSpaceAdapter(ArrowSpaceAdapter):
    def __init__(self, module: Any, cache_size: int = 8) -> None:
        super().__init__(available=True, backend="arrowspace")
        self._mod = module
        self._cache = _LRUIndexCache(maxsize=cache_size)

    @staticmethod
    def _slug(dataset_id: str) -> str:
        return dataset_id.replace("/", "__").replace("\\", "__")

    def _persist_csr(self, index_store: Path, slug: str, gl: Any, meta: dict[str, Any]) -> None:
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
                z = zarr.open(str(zarr_path), mode="w", shape=arr_val.shape,
                              dtype=arr_val.dtype, chunks=True, zarr_format=3)
                z[:] = arr_val
            meta_dict = dict(meta)
            meta_dict["csr_shape"] = list(csr_shape)
            (dest / "meta.json").write_text(json.dumps(meta_dict))
            log.info("Persisted graph-Laplacian CSR to %s", dest)
        except Exception:
            log.warning("Failed to persist CSR for '%s'", slug, exc_info=True)

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gp = graph_params or DEFAULT_GRAPH_PARAMS
        arr64 = np.asarray(array, dtype=np.float64)
        if arr64.ndim != 2:
            raise ValueError(
                f"arrowspace requires a 2-D array (items x features); got shape {arr64.shape}"
            )
        log.info("Building index for '%s' shape=%s params=%s", dataset_id, arr64.shape, gp)
        aspace, gl = self._mod.ArrowSpaceBuilder().build(gp, arr64)
        entry = _IndexEntry(
            aspace=aspace, gl=gl,
            nitems=int(aspace.nitems),
            nfeatures=int(aspace.nfeatures),
            nclusters=int(aspace.nclusters),
        )
        self._cache.put(dataset_id, entry)
        meta = {"nitems": entry.nitems, "nfeatures": entry.nfeatures, "nclusters": entry.nclusters}
        self._persist_csr(index_store, self._slug(dataset_id), gl, meta)
        return meta

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No index built for '{dataset_id}'. "
                "Call POST /api/datasets/{id}/index first."
            )
        return entry

    # ------------------------------------------------------------------
    # Helpers shared by search methods
    # ------------------------------------------------------------------

    def _vec(self, query: dict[str, Any]) -> np.ndarray:
        """Extract and validate 'vector' from query dict, return float64 array.

        Raises HTTPException(422) if the value cannot be coerced to float64.
        Raises MetadataUnavailable(404) if 'vector' key is absent.
        """
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable("'vector' is required in search body")
        try:
            return np.asarray(vec, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"'vector' must be a list of numbers; got: {type(vec).__name__}",
            ) from exc

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        lam = list(entry.aspace.lambdas())
        lam_sorted = [[float(v), int(i)] for v, i in entry.aspace.lambdas_sorted()]
        return {
            "nitems": entry.nitems,
            "lambdas": [float(v) for v in lam],
            "lambdas_sorted": lam_sorted,
        }

    def graph_laplacian_info(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        nnodes = int(entry.gl.nnodes)
        # gl.shape may be a property/method returning a non-iterable on some
        # arrowspace versions; guard with try/except and fall back to (nnodes, nnodes).
        try:
            gl_shape = list(entry.gl.shape)
        except TypeError:
            gl_shape = [nnodes, nnodes]
        return {
            "nnodes": nnodes,
            "shape": gl_shape,
            "graph_params": entry.gl.graph_params,
        }

    def get_item(self, dataset_id: str, idx: int) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        if idx < 0 or idx >= entry.nitems:
            raise HTTPException(
                status_code=404,
                detail=f"Item index {idx} out of range [0, {entry.nitems}).",
            )
        vec = entry.aspace.get_item(idx)
        return {
            "item_index": idx,
            "vector": [float(v) for v in vec],
        }

    def get_all_items(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        items = entry.aspace.get_all_items()
        return {
            "nitems": entry.nitems,
            "items": [[float(v) for v in row] for row in items],
        }

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        q_arr = self._vec(query)
        tau = float(query.get("tau", 1.0))
        hits = entry.aspace.search(q_arr, entry.gl, tau)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    def search_batch(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vecs_raw = query.get("vectors")
        if vecs_raw is None:
            raise MetadataUnavailable("'vectors' is required in search_batch body")
        try:
            vecs = np.asarray(vecs_raw, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="'vectors' must be a 2-D list of numbers") from exc
        tau = float(query.get("tau", 1.0))
        batch_hits = entry.aspace.search_batch(vecs, entry.gl, tau)
        return {
            "backend": "arrowspace",
            "results": [
                [{"index": int(i), "score": float(s)} for i, s in hits]
                for hits in batch_hits
            ],
        }

    def search_energy(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        q_arr = self._vec(query)
        k = int(query.get("k", DEFAULT_SEARCH_K))
        hits = entry.aspace.search_energy(q_arr, entry.gl, k)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    def search_hybrid(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        q_arr = self._vec(query)
        alpha = float(query.get("alpha", 0.5))
        hits = entry.aspace.search_hybrid(q_arr, entry.gl, alpha)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    def search_linear_sorted(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        q_arr = self._vec(query)
        k = int(query.get("k", DEFAULT_SEARCH_K))
        hits = entry.aspace.search_linear_sorted(q_arr, entry.gl, k)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    def _spot_hits(self, hits: Any) -> list[dict[str, Any]]:
        return [{"index": int(i), "score": float(s)} for i, s in hits]

    def spot_motives_eigen(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {"method": "spot_motives_eigen", "results": self._spot_hits(entry.aspace.spot_motives_eigen())}

    def spot_motives_energy(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {"method": "spot_motives_energy", "results": self._spot_hits(entry.aspace.spot_motives_energy())}

    def spot_subg_centroids(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {"method": "spot_subg_centroids", "results": self._spot_hits(entry.aspace.spot_subg_centroids())}

    def spot_subg_motives(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {"method": "spot_subg_motives", "results": self._spot_hits(entry.aspace.spot_subg_motives())}

    def manifold_data(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        lam_sorted = [[float(v), int(i)] for v, i in entry.aspace.lambdas_sorted()]
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "lambdas_sorted": lam_sorted[:50],
        }

    def stats_data(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        nnodes = int(entry.gl.nnodes)
        # gl.shape may be a non-iterable built-in on some arrowspace versions;
        # guard with try/except and fall back to (nnodes, nnodes).
        try:
            gl_shape = list(entry.gl.shape)
        except TypeError:
            gl_shape = [nnodes, nnodes]
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "gl_nodes": nnodes,
            "gl_shape": gl_shape,
        }

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "manifold.json")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "stats.json")

    def sidecar_search(self, dataset_path: Path, q: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return _SidecarAdapter().sidecar_search(dataset_path, q, limit=limit)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load() -> ArrowSpaceAdapter:
    """Return the best available ArrowSpace adapter.

    Priority:
    1. arrowspace package importable  -> _ArrowSpaceAdapter
    2. fallback                       -> _SidecarAdapter

    FIX: broadened except to catch Exception (not just ImportError) because
    the installed arrowspace package raises NameError in __init__.py when its
    internal submodule reference fails — that is not an ImportError and was
    previously crashing the server instead of gracefully falling back.
    """
    from .settings import get_settings

    try:
        import arrowspace as _mod  # type: ignore
        cache_size = get_settings().index_cache_size
        log.info("arrowspace package found; using live adapter (cache_size=%d)", cache_size)
        return _ArrowSpaceAdapter(_mod, cache_size=cache_size)
    except Exception:  # catches ImportError AND NameError from broken __init__
        log.info("arrowspace package not available; using sidecar adapter")
        return _SidecarAdapter()


def reset_adapter_cache() -> None:
    # Guard: load() may be monkey-patched in tests to a plain function
    # that does not have .cache_clear(). Only call it when present.
    if hasattr(load, "cache_clear"):
        load.cache_clear()
