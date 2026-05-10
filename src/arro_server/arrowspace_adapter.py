"""Adapter for arrowspace / ArrowSpace graph-Laplacian index.

The ``arrowspace`` package (pip install arrowspace,
repo: https://github.com/tuned-org-uk/pyarrowspace) may not be available in
every environment.  We never import it at module load — :func:`load` attempts
the import lazily and returns a stub adapter that falls back to the sidecar
JSON adapter.

Real arrowspace API (confirmed from package introspection)::

    from arrowspace import ArrowSpaceBuilder

    # Build index — returns a single ArrowSpace object (NOT a tuple)
    aspace = ArrowSpaceBuilder().build_and_store(graph_params, items)

    # GraphLaplacian is an attribute on the returned object
    gl = aspace.gl

    # Reload persisted index without recomputing
    aspace = arrowspace.load_arrowspace(
        storage_path="storage/",
        dataset_name="dataset_{uuid}",
        graph_params={...},
        energy=False,
    )
    gl = aspace.gl

ArrowSpace object public surface::

    aspace.nitems          int
    aspace.nfeatures       int
    aspace.nclusters       int
    aspace.gl              GraphLaplacian
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
  1. Calls ``ArrowSpaceBuilder().build_and_store(graph_params, items)``.
  2. Persists the GraphLaplacian as Zarr v3 CSR arrays under
     ``<index_store>/<slug>/`` via ``_persist_csr()``.
  3. Records ``{dataset_id -> slug}`` in
     ``<index_store>/index_manifest.json``.

On ``load()`` (server startup) the adapter:
  1. Reads ``index_manifest.json`` (if present).
  2. Calls ``arrowspace.load_arrowspace()`` for every known entry and
     pre-populates the LRU cache — indices survive restarts without
     recomputing.
"""

from __future__ import annotations

import json
import logging
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

    def delete_index(self, dataset_id, index_store):
        raise OptionalDependencyMissing("arrowspace", "delete_index")

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
        """Filesystem-safe slug derived from dataset_id."""
        return "dataset_" + dataset_id.replace("/", "__").replace("\\", "__")

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
            (dest / "meta.json").write_text(json.dumps(meta_dict))
            log.info("Persisted graph-Laplacian CSR to %s", dest)
        except Exception:
            log.warning("Failed to persist CSR for '%s'", slug, exc_info=True)

    # ------------------------------------------------------------------
    # build_index
    # ------------------------------------------------------------------

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

        slug = self._slug(dataset_id)
        index_store.mkdir(parents=True, exist_ok=True)

        log.info(
            "Building index for '%s' shape=%s params=%s",
            dataset_id, arr64.shape, gp,
        )

        # Correct API: build_and_store(graph_params, items) -> ArrowSpace
        # No storage_path / dataset_name kwargs — those do not exist.
        # The GraphLaplacian is accessed as aspace.gl (not a second return value).
        aspace = self._mod.ArrowSpaceBuilder().build_and_store(gp, arr64)
        gl = aspace.gl

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

        meta = {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
        }
        # Persist CSR Zarr arrays and update manifest
        self._persist_csr(index_store, slug, gl, meta)
        _Manifest(index_store).put(dataset_id, slug)
        log.info("Index for '%s' persisted as '%s'", dataset_id, slug)

        return {**meta, "slug": slug, "graph_params": gp}

    # ------------------------------------------------------------------
    # Auto-load persisted indices at startup
    # ------------------------------------------------------------------

    def load_persisted(self, index_store: Path) -> int:
        """Read manifest and pre-load all known indices into the LRU cache.

        Called once at server startup. Returns the number of indices loaded.
        Failures for individual indices are logged and skipped.
        """
        manifest = _Manifest(index_store).get_all()
        if not manifest:
            log.info("No persisted indices found in %s", index_store)
            return 0

        loaded = 0
        for dataset_id, slug in manifest.items():
            if dataset_id in self._cache:
                log.debug("'%s' already in cache, skipping reload", dataset_id)
                loaded += 1
                continue
            try:
                log.info("Loading persisted index '%s' from slug '%s'", dataset_id, slug)
                aspace = self._mod.load_arrowspace(
                    storage_path=str(index_store),
                    dataset_name=slug,
                    graph_params=DEFAULT_GRAPH_PARAMS,
                    energy=False,
                )
                gl = aspace.gl
                entry = _IndexEntry(
                    aspace=aspace,
                    gl=gl,
                    nitems=int(aspace.nitems),
                    nfeatures=int(aspace.nfeatures),
                    nclusters=int(aspace.nclusters),
                    slug=slug,
                    graph_params=DEFAULT_GRAPH_PARAMS,
                )
                self._cache.put(dataset_id, entry)
                loaded += 1
                log.info("Loaded '%s' (%d items)", dataset_id, entry.nitems)
            except Exception:
                log.warning(
                    "Failed to load persisted index for '%s'; skipping",
                    dataset_id,
                    exc_info=True,
                )

        return loaded

    # ------------------------------------------------------------------
    # Delete index
    # ------------------------------------------------------------------

    def delete_index(self, dataset_id: str, index_store: Path) -> bool:
        manifest = _Manifest(index_store)
        slug = manifest.remove(dataset_id)
        cache_hit = self._cache.delete(dataset_id)

        if slug:
            csr_dir = index_store / slug
            if csr_dir.exists():
                import shutil
                shutil.rmtree(csr_dir, ignore_errors=True)
                log.info("Deleted CSR Zarr files for '%s' at %s", dataset_id, csr_dir)

        return bool(slug or cache_hit)

    # ------------------------------------------------------------------
    # Health / introspection
    # ------------------------------------------------------------------

    def indexed_datasets(self) -> list[str]:
        return self._cache.keys()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No index built for '{dataset_id}'. "
                "Call POST /api/datasets/{id}/index first."
            )
        return entry

    def _vec(self, query: dict[str, Any]) -> np.ndarray:
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
            raise HTTPException(
                status_code=422, detail="'vectors' must be a 2-D list of numbers"
            ) from exc
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

    def sidecar_search(
        self, dataset_path: Path, q: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        return _SidecarAdapter().sidecar_search(dataset_path, q, limit=limit)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load() -> ArrowSpaceAdapter:
    """Return the best available ArrowSpace adapter and pre-load persisted indices.

    Priority:
    1. arrowspace package importable  -> _ArrowSpaceAdapter (with auto-load)
    2. fallback                       -> _SidecarAdapter
    """
    from .settings import get_settings

    try:
        import arrowspace as _mod  # type: ignore
        settings = get_settings()
        cache_size = settings.index_cache_size
        index_store = settings.effective_index_store()
        log.info("arrowspace package found; using live adapter (cache_size=%d)", cache_size)
        adapter = _ArrowSpaceAdapter(_mod, cache_size=cache_size)
        n = adapter.load_persisted(index_store)
        if n:
            log.info("Pre-loaded %d persisted index(es) from %s", n, index_store)
        return adapter
    except Exception:  # catches ImportError AND NameError from broken __init__
        log.info("arrowspace package not available; using sidecar adapter")
        return _SidecarAdapter()


def reset_adapter_cache() -> None:
    if hasattr(load, "cache_clear"):
        load.cache_clear()
