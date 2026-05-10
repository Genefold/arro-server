"""Adapter for arrowspace / ArrowSpace graph-Laplacian index.

The ``arrowspace`` package (pip install arrowspace,
repo: https://github.com/tuned-org-uk/pyarrowspace) may not be available in
every environment.  We never import it at module load — :func:`load` attempts
the import lazily and returns a stub adapter that falls back to the sidecar
JSON adapter.

Real arrowspace API (confirmed from package introspection)::

    from arrowspace import ArrowSpaceBuilder

    # Build index — returns a 2-tuple (ArrowSpace, GraphLaplacian)
    aspace, gl = ArrowSpaceBuilder().build_and_store(graph_params, items)

    # Reload persisted index without recomputing — also returns a 2-tuple
    aspace, gl = arrowspace.load_arrowspace(
        storage_path="storage/",
        dataset_name="dataset_{uuid}",
        graph_params={...},
        energy=False,
    )

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

Persistence
-----------
On every ``build_index()`` call the adapter:
  1. Calls ``ArrowSpaceBuilder().build_and_store(graph_params, items)``
     which returns ``(aspace, gl)`` — a 2-tuple.
  2. Persists the GraphLaplacian as Zarr v3 CSR arrays under
     ``<index_store>/<slug>/`` via ``_persist_csr()``.
  3. Records ``{dataset_id -> slug}`` in
     ``<index_store>/index_manifest.json``.

On ``load()`` (server startup) the adapter:
  1. Reads ``index_manifest.json`` (if present).
  2. Calls ``arrowspace.load_arrowspace()`` for every known entry —
     also returns ``(aspace, gl)`` — and pre-populates the LRU cache.
     Indices survive restarts without recomputing.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
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

MANIFEST_FILENAME = "index_manifest.json"


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
    def delete_index(self, dataset_id: str, index_store: Path) -> bool: ...

    @abstractmethod
    def indexed_datasets(self) -> list[str]: ...

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

    # These are implemented only in _ArrowSpaceAdapter but declared here
    # so routes can call them via isinstance check.
    def manifold_data(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def stats_data(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def graph_laplacian_info(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def get_item(self, dataset_id: str, idx: int) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def get_all_items(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def search_batch(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def search_energy(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def search_hybrid(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def search_linear_sorted(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def spot_motives_eigen(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def spot_motives_energy(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def spot_subg_centroids(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError

    def spot_subg_motives(self, dataset_id: str) -> dict[str, Any]:  # type: ignore[return]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Manifest helper (thread-safe)
# ---------------------------------------------------------------------------


class _Manifest:
    """JSON file that maps dataset_id -> slug (CSR Zarr directory name).

    Thread-safe: all reads and writes are protected by a lock.
    """

    def __init__(self, index_store: Path) -> None:
        self._path = index_store / MANIFEST_FILENAME
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            log.warning("Could not read manifest at %s", self._path, exc_info=True)
            return {}

    def _write(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    def get_all(self) -> dict[str, str]:
        with self._lock:
            return dict(self._read())

    def put(self, dataset_id: str, slug: str) -> None:
        with self._lock:
            data = self._read()
            data[dataset_id] = slug
            self._write(data)

    def remove(self, dataset_id: str) -> str | None:
        """Remove entry; return the slug that was stored, or None."""
        with self._lock:
            data = self._read()
            slug = data.pop(dataset_id, None)
            if slug is not None:
                self._write(data)
            return slug


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

    def delete_index(self, dataset_id: str, index_store: Path) -> bool:
        raise OptionalDependencyMissing("arrowspace", "delete_index")

    def indexed_datasets(self) -> list[str]:
        return []

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

    def delete_index(self, dataset_id, index_store) -> bool:
        # No index can exist when arrowspace is unavailable — return False gracefully
        # rather than raising, so DELETE /index always returns 200 {"deleted": false}.
        return False

    def indexed_datasets(self) -> list[str]:
        return []

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
    slug: str = field(default="")
    graph_params: dict[str, Any] = field(default_factory=dict)


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

    def keys(self) -> list[str]:
        return list(self._data.keys())

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
        """Filesystem-safe slug derived from dataset_id (no prefix)."""
        return dataset_id.replace("/", "__").replace("\\", "__")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No ArrowSpace index in cache for '{dataset_id}'. "
                "Call POST /index first."
            )
        return entry

    def _hits_to_results(self, hits: Any) -> list[dict[str, Any]]:
        """Normalise a list of (index, score) tuples to dicts."""
        results = []
        for item in hits:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                results.append({"index": int(item[0]), "score": float(item[1])})
            else:
                results.append({"index": int(item), "score": 0.0})
        return results

    # ------------------------------------------------------------------
    # Persist GraphLaplacian as Zarr v3 CSR arrays
    # ------------------------------------------------------------------

    def _persist_csr(self, index_store: Path, slug: str, gl: Any, meta: dict[str, Any]) -> None:
        try:
            import zarr  # type: ignore
        except ImportError:
            log.warning("zarr not installed; graph-Laplacian CSR will not be persisted")
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
                    str(zarr_path), mode="w",
                    shape=arr_val.shape, dtype=arr_val.dtype,
                    chunks=True, zarr_format=3,
                )
                z[:] = arr_val
            meta_dict = dict(meta)
            meta_dict["csr_shape"] = list(csr_shape)
            (dest / "meta.json").write_text(json.dumps(meta_dict, indent=2))
            log.info("Persisted CSR GraphLaplacian to %s", dest)
        except Exception:
            log.warning("Failed to persist CSR arrays for slug '%s'", slug, exc_info=True)

    # ------------------------------------------------------------------
    # Build / delete index
    # ------------------------------------------------------------------

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if array.ndim != 2:
            raise ValueError(
                f"ArrowSpace requires a 2-D array; got shape {array.shape}. "
                "Please pass a 2-D (rows x features) float64 array."
            )
        gp = dict(graph_params or DEFAULT_GRAPH_PARAMS)
        arr64 = array.astype(np.float64)
        # Phase 2 contract: build_and_store(graph_params, array) -> (aspace, gl)
        aspace, gl = self._mod.ArrowSpaceBuilder().build_and_store(gp, arr64)
        slug = self._slug(dataset_id)
        meta: dict[str, Any] = {
            "nitems": int(aspace.nitems),
            "nfeatures": int(aspace.nfeatures),
            "nclusters": int(aspace.nclusters),
        }
        entry = _IndexEntry(
            aspace=aspace,
            gl=gl,
            nitems=int(aspace.nitems),
            nfeatures=int(aspace.nfeatures),
            nclusters=int(aspace.nclusters),
            slug=slug,
            graph_params=gp,
        )
        self._cache.put(dataset_id, entry)
        self._persist_csr(index_store, slug, gl, meta)
        _Manifest(index_store).put(dataset_id, slug)
        return meta

    def delete_index(self, dataset_id: str, index_store: Path) -> bool:
        in_cache = self._cache.delete(dataset_id)
        slug = _Manifest(index_store).remove(dataset_id)
        if slug:
            dest = index_store / slug
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
                log.info("Deleted CSR directory %s", dest)
        return in_cache or bool(slug)

    def indexed_datasets(self) -> list[str]:
        return self._cache.keys()

    # ------------------------------------------------------------------
    # Load persisted indices on startup
    # ------------------------------------------------------------------

    def load_persisted(self, index_store: Path) -> None:
        """Reload indices from disk into the LRU cache (called at startup)."""
        manifest = _Manifest(index_store).get_all()
        for dataset_id, slug in manifest.items():
            dest = index_store / slug
            meta_file = dest / "meta.json"
            if not meta_file.exists():
                log.warning("Missing meta.json for '%s' at %s — skipping", dataset_id, dest)
                continue
            try:
                meta = json.loads(meta_file.read_text())
                aspace, gl = self._mod.load_arrowspace(
                    storage_path=str(index_store),
                    dataset_name=slug,
                    graph_params=meta.get("graph_params", DEFAULT_GRAPH_PARAMS),
                    energy=False,
                )
                entry = _IndexEntry(
                    aspace=aspace,
                    gl=gl,
                    nitems=int(aspace.nitems),
                    nfeatures=int(aspace.nfeatures),
                    nclusters=int(aspace.nclusters),
                    slug=slug,
                    graph_params=meta.get("graph_params", {}),
                )
                self._cache.put(dataset_id, entry)
                log.info("Reloaded ArrowSpace index for '%s' from %s", dataset_id, dest)
            except Exception:
                log.warning(
                    "Failed to reload index for '%s' from %s", dataset_id, dest, exc_info=True
                )

    # ------------------------------------------------------------------
    # Lambdas
    # ------------------------------------------------------------------

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        raw = entry.aspace.lambdas()
        lam_list = [float(v) for v in raw]
        sorted_raw = entry.aspace.lambdas_sorted()
        lam_sorted = [[float(v), int(i)] for v, i in sorted_raw]
        return {
            "nitems": entry.nitems,
            "lambdas": lam_list,
            "lambdas_sorted": lam_sorted,
        }

    # ------------------------------------------------------------------
    # Manifold and Stats (live)
    # ------------------------------------------------------------------

    def manifold_data(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        sorted_raw = entry.aspace.lambdas_sorted()
        lam_sorted = [[float(v), int(i)] for v, i in sorted_raw][:50]
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "lambdas_sorted": lam_sorted,
        }

    def stats_data(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        gl_shape = list(entry.gl.shape) if hasattr(entry.gl.shape, "__iter__") else [entry.gl.shape, entry.gl.shape]
        return {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
            "gl_nodes": int(entry.gl.nnodes),
            "gl_shape": [int(x) for x in gl_shape],
        }

    # ------------------------------------------------------------------
    # Graph Laplacian info
    # ------------------------------------------------------------------

    def graph_laplacian_info(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        gl_shape = list(entry.gl.shape) if hasattr(entry.gl.shape, "__iter__") else [entry.gl.shape, entry.gl.shape]
        return {
            "nnodes": int(entry.gl.nnodes),
            "shape": [int(x) for x in gl_shape],
            "graph_params": dict(entry.gl.graph_params),
        }

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------

    def get_item(self, dataset_id: str, idx: int) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        raw = entry.aspace.get_item(idx)
        if hasattr(raw, "tolist"):
            vec = raw.tolist()
        else:
            vec = [float(v) for v in raw]
        return {"item_index": idx, "vector": vec}

    def get_all_items(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        raw = entry.aspace.get_all_items()
        if hasattr(raw, "tolist"):
            items = raw.tolist()
        else:
            items = [[float(v) for v in row] for row in raw]
        return {"nitems": entry.nitems, "items": items}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable("Missing required field: vector")
        tau = float(query.get("tau", 1.0))
        hits = entry.aspace.search(np.asarray(vec, dtype=np.float64), entry.gl, tau)
        return {"backend": "arrowspace", "results": self._hits_to_results(hits)}

    def search_batch(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vecs = query.get("vectors")
        if vecs is None:
            raise MetadataUnavailable("Missing required field: vectors")
        tau = float(query.get("tau", 1.0))
        batch = np.asarray(vecs, dtype=np.float64)
        hits_batch = entry.aspace.search_batch(batch, entry.gl, tau)
        return {
            "backend": "arrowspace",
            "results": [self._hits_to_results(hits) for hits in hits_batch],
        }

    def search_energy(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable("Missing required field: vector")
        k = int(query.get("k", DEFAULT_SEARCH_K))
        hits = entry.aspace.search_energy(np.asarray(vec, dtype=np.float64), entry.gl, k)
        return {"backend": "arrowspace", "results": self._hits_to_results(hits)}

    def search_hybrid(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable("Missing required field: vector")
        alpha = float(query.get("alpha", 0.5))
        hits = entry.aspace.search_hybrid(np.asarray(vec, dtype=np.float64), entry.gl, alpha)
        return {"backend": "arrowspace", "results": self._hits_to_results(hits)}

    def search_linear_sorted(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable("Missing required field: vector")
        k = int(query.get("k", DEFAULT_SEARCH_K))
        hits = entry.aspace.search_linear_sorted(np.asarray(vec, dtype=np.float64), entry.gl, k)
        return {"backend": "arrowspace", "results": self._hits_to_results(hits)}

    # ------------------------------------------------------------------
    # Spot methods
    # ------------------------------------------------------------------

    def _spot(self, dataset_id: str, method_name: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        fn = getattr(entry.aspace, method_name)
        hits = fn()
        return {"method": method_name, "results": self._hits_to_results(hits)}

    def spot_motives_eigen(self, dataset_id: str) -> dict[str, Any]:
        return self._spot(dataset_id, "spot_motives_eigen")

    def spot_motives_energy(self, dataset_id: str) -> dict[str, Any]:
        return self._spot(dataset_id, "spot_motives_energy")

    def spot_subg_centroids(self, dataset_id: str) -> dict[str, Any]:
        return self._spot(dataset_id, "spot_subg_centroids")

    def spot_subg_motives(self, dataset_id: str) -> dict[str, Any]:
        return self._spot(dataset_id, "spot_subg_motives")

    # ------------------------------------------------------------------
    # Sidecar pass-through (no-ops for the live adapter)
    # ------------------------------------------------------------------

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        # Live adapter doesn't read sidecars; routes try live first anyway.
        raise MetadataUnavailable("sidecar not used by live ArrowSpace adapter")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        raise MetadataUnavailable("sidecar not used by live ArrowSpace adapter")

    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        # Fall back to sidecar search for keyword queries
        return _SidecarAdapter().sidecar_search(dataset_path, q, limit=limit)


# ---------------------------------------------------------------------------
# Module-level load() + reset_adapter_cache()
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load() -> ArrowSpaceAdapter:
    """Lazily import arrowspace and return the appropriate adapter.

    Falls back to _SidecarAdapter if the package raises any Exception on import.
    Falls back to _UnavailableAdapter if neither live nor sidecar is possible.
    """
    try:
        import arrowspace as _mod  # type: ignore
        # Verify the module is usable
        _ = _mod.ArrowSpaceBuilder
        log.info("arrowspace package loaded; using live ArrowSpace adapter")
        return _ArrowSpaceAdapter(_mod)
    except Exception:
        log.info(
            "arrowspace package not available or broken; falling back to sidecar adapter",
            exc_info=True,
        )
        return _SidecarAdapter()


def reset_adapter_cache() -> None:
    """Clear the module-level load() LRU cache (used in tests to reset state)."""
    if hasattr(load, "cache_clear"):
        load.cache_clear()
