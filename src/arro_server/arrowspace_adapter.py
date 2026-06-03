"""Adapter for arrowspace / ArrowSpace graph-Laplacian index.

The ``arrowspace`` package (pip install arrowspace,
repo: https://github.com/tuned-org-uk/pyarrowspace) may not be available in
every environment.  We never import it at module load — :func:`load` attempts
the import lazily and returns a stub adapter that falls back to the sidecar
JSON adapter.

Real arrowspace API (confirmed from package introspection)::

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

Persistence (Phase 2)
---------------------
After ``build_index()`` the adapter maintains a JSON manifest at
``{index_store}/index_manifest.json`` that maps ``dataset_id`` to its
``{dataset_name, graph_params}`` metadata.  On adapter startup
``reload_from_manifest()`` is called from the FastAPI lifespan hook to
restore all previously-built indices into the LRU cache without recomputing.

The persistence format uses ``arrowspace.load_arrowspace()`` which reads the
Parquet files written by the Rust builder's ``with_persistence`` option.
The dataset_name is generated as ``{slug}_{uuid}`` the first time an index is
built, then recorded in the manifest so the same name is used on reload.

Note on ``build_and_store()`` vs ``build()``:
  ``ArrowSpaceBuilder.build_and_store()`` hardcodes the output directory to
  ``CWD/storage/`` inside the Rust code and cannot be redirected.  We
  therefore use ``build()`` for the in-memory result and separately invoke
  the builder with ``with_persistence(dir_path, dataset_name)`` set to the
  configured ``index_store``, so the output respects the server's settings.
"""

from __future__ import annotations

import json
import logging
import uuid
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
    def has_index(self, dataset_id: str) -> bool:
        """Return True if dataset_id has a built index available in the cache."""
        ...

    @abstractmethod
    def indexed_datasets(self) -> list[str]: ...

    @abstractmethod
    def reload_from_manifest(self, index_store: Path) -> list[str]: ...

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

    @abstractmethod
    def graph_export(self, dataset_id: str, fmt: str) -> dict[str, Any]: ...

    @abstractmethod
    def spectral_metrics(self, dataset_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def motives(self, dataset_id: str, mode: str) -> dict[str, Any]: ...

    @abstractmethod
    def subgraphs(self, dataset_id: str, mode: str) -> dict[str, Any]: ...

    @abstractmethod
    def search_with_mode(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]: ...


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

    def has_index(self, dataset_id: str) -> bool:
        return False

    def indexed_datasets(self) -> list[str]:
        return []

    def reload_from_manifest(self, index_store: Path) -> list[str]:
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

    def graph_export(self, dataset_id: str, fmt: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "graph_export")

    def spectral_metrics(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "spectral_metrics")

    def motives(self, dataset_id: str, mode: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "motives")

    def subgraphs(self, dataset_id: str, mode: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "subgraphs")

    def search_with_mode(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "search_with_mode")


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

    def has_index(self, dataset_id: str) -> bool:
        return False

    def indexed_datasets(self) -> list[str]:
        return []

    def reload_from_manifest(self, index_store: Path) -> list[str]:
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

    def graph_export(self, dataset_id, fmt):
        raise OptionalDependencyMissing("arrowspace", "graph_export")

    def spectral_metrics(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "spectral_metrics")

    def motives(self, dataset_id, mode):
        raise OptionalDependencyMissing("arrowspace", "motives")

    def subgraphs(self, dataset_id, mode):
        raise OptionalDependencyMissing("arrowspace", "subgraphs")

    def search_with_mode(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search_with_mode")


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

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Manifest helpers (module-level, pure functions)
# ---------------------------------------------------------------------------


def _read_manifest(index_store: Path) -> dict[str, Any]:
    """Read index_manifest.json; return empty dict if absent or corrupt."""
    path = index_store / MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Could not read manifest at %s; treating as empty", path)
        return {}


def _write_manifest(index_store: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the manifest JSON."""
    index_store.mkdir(parents=True, exist_ok=True)
    path = index_store / MANIFEST_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


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

    @staticmethod
    def _new_dataset_name(slug: str) -> str:
        """Generate a unique arrowspace dataset name for a given slug."""
        return f"{slug}_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # ABC: has_index
    # ------------------------------------------------------------------

    def has_index(self, dataset_id: str) -> bool:
        """Return True if dataset_id has a built index in the LRU cache."""
        return dataset_id in self._cache

    def _persist_csr(self, index_store: Path, slug: str, gl: Any, meta: dict[str, Any]) -> None:
        """Persist GraphLaplacian CSR arrays as Zarr for quick inspection/export.

        This is supplementary to the Parquet persistence used for reload;
        it writes the raw CSR matrices so external tools can read them without
        the arrowspace package.
        """
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
            log.warning("Failed to persist CSR for '%s'", slug, exc_info=True)

    def _build_with_persistence(
        self,
        index_store: Path,
        dataset_name: str,
        array: np.ndarray,
        graph_params: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Build ArrowSpace index and persist Parquet in a single call.

        Uses the builder's ``with_persistence`` option so that the on-disk
        files are guaranteed identical to the in-memory result.  Falls back
        to a bare ``build()`` when the installed arrowspace version does
        not expose ``with_persistence`` (Phase 1 behaviour — index lives
        in memory only).

        We cannot use ``build_and_store()`` directly because its Rust
        implementation hardcodes the output path to ``CWD/storage/``.
        Instead we configure the builder with ``with_persistence(dir, name)``
        before calling ``build()``, which triggers the same Parquet write path
        but respects the configured ``index_store``.

        Returns the ``(aspace, gl)`` tuple from the single build call.
        """
        rows: list[list[float]] = array.tolist()
        try:
            builder = self._mod.ArrowSpaceBuilder()
            gp = graph_params
            if hasattr(builder, "with_lambda_graph") and hasattr(builder, "with_persistence"):
                builder = (
                    builder.with_lambda_graph(gp["eps"], gp["k"], gp["topk"], gp["p"], gp["sigma"])
                    .with_sparsity_check(False)
                    .with_persistence(str(index_store), dataset_name)
                )
                aspace, gl = builder.build(rows)
                log.info(
                    "Persisted ArrowSpace Parquet for '%s' at %s",
                    dataset_name,
                    index_store,
                )
                return aspace, gl

            log.warning(
                "ArrowSpaceBuilder does not expose with_persistence; "
                "index for '%s' will not survive restart",
                dataset_name,
            )
            return builder.build(gp, array)
        except Exception:
            log.warning(
                "Failed to persist Parquet for '%s'; building in memory only",
                dataset_name,
                exc_info=True,
            )
            return self._mod.ArrowSpaceBuilder().build(graph_params, array)

    # ------------------------------------------------------------------
    # Phase 2 — public persistence API
    # ------------------------------------------------------------------

    def reload_from_manifest(self, index_store: Path) -> list[str]:
        """Load all indices recorded in the manifest into the LRU cache.

        Called from the FastAPI lifespan hook on server startup.  Safe to call
        when the manifest is absent (returns empty list) or when individual
        entries cannot be loaded (logs a warning, skips that entry).

        Each manifest entry MUST have a ``dataset_name`` key (written by
        ``build_index``).  Entries missing this key are unrecoverable — the
        UUID suffix appended by ``_new_dataset_name`` means the bare slug will
        never match the Parquet files on disk.  Such entries are skipped with
        a warning rather than silently falling back to an incorrect path.
        """
        manifest = _read_manifest(index_store)
        if not manifest:
            log.info("No index manifest found at %s; starting with empty cache", index_store)
            return []
        loaded: list[str] = []
        for dataset_id, info in manifest.items():
            dataset_name = info.get("dataset_name")
            if not dataset_name:
                log.warning(
                    "Manifest entry for '%s' is missing 'dataset_name'; "
                    "cannot reload — skipping (entry is unrecoverable without the UUID suffix)",
                    dataset_id,
                )
                continue
            graph_params = info.get("graph_params", DEFAULT_GRAPH_PARAMS)
            try:
                aspace, gl = self._mod.load_arrowspace(
                    storage_path=str(index_store),
                    dataset_name=dataset_name,
                    graph_params=graph_params,
                    energy=False,
                )
                entry = _IndexEntry(
                    aspace=aspace,
                    gl=gl,
                    nitems=int(aspace.nitems),
                    nfeatures=int(aspace.nfeatures),
                    nclusters=int(aspace.nclusters),
                )
                self._cache.put(dataset_id, entry)
                loaded.append(dataset_id)
                log.info("Reloaded index for '%s' from %s", dataset_id, index_store)
            except Exception:
                log.warning(
                    "Could not reload index for '%s' (dataset_name='%s'); skipping",
                    dataset_id,
                    dataset_name,
                    exc_info=True,
                )
        return loaded

    def delete_index(self, dataset_id: str, index_store: Path) -> bool:
        """Remove dataset_id from cache, Parquet files, CSR Zarr, and manifest.

        Returns True if the entry existed (either in cache or manifest),
        False if nothing was found.
        """
        manifest = _read_manifest(index_store)
        existed = self._cache.delete(dataset_id) or (dataset_id in manifest)

        # Remove Parquet files written by arrowspace builder
        info = manifest.pop(dataset_id, {})
        dataset_name = info.get("dataset_name", "")
        if dataset_name:
            for suffix in ("_items.parquet", "_graph.parquet"):
                f = index_store / f"{dataset_name}{suffix}"
                if f.exists():
                    try:
                        f.unlink()
                        log.info("Deleted %s", f)
                    except Exception:
                        log.warning("Could not delete %s", f, exc_info=True)

        # Remove CSR Zarr directory
        slug = self._slug(dataset_id)
        csr_dir = index_store / slug
        if csr_dir.exists():
            import shutil

            try:
                shutil.rmtree(str(csr_dir))
                log.info("Deleted CSR Zarr directory %s", csr_dir)
            except Exception:
                log.warning("Could not delete CSR dir %s", csr_dir, exc_info=True)

        # Always write the updated manifest when the entry existed, regardless
        # of the on-disk state.  The previous guard re-read the manifest after
        # pop(), so the on-disk copy still contained the entry, making the
        # condition almost always True — but in the wrong direction.  This is
        # simpler and always correct.
        if existed:
            _write_manifest(index_store, manifest)

        return existed

    def indexed_datasets(self) -> list[str]:
        """Return the list of dataset IDs currently in the LRU cache."""
        return self._cache.keys()

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
        # Re-use the existing dataset_name from the manifest so that rebuilding
        # a dataset replaces the same Parquet files on disk.
        manifest = _read_manifest(index_store)
        existing = manifest.get(dataset_id, {})
        dataset_name = existing.get("dataset_name") or self._new_dataset_name(slug)

        log.info(
            "Building index for '%s' (dataset_name='%s') shape=%s params=%s",
            dataset_id,
            dataset_name,
            arr64.shape,
            gp,
        )

        # Single build: persist Parquet AND get the in-memory result in one call
        aspace, gl = self._build_with_persistence(index_store, dataset_name, arr64, gp)

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

        # Persist CSR (supplementary, for export/inspection)
        self._persist_csr(index_store, slug, gl, meta)

        # Update manifest
        manifest[dataset_id] = {"dataset_name": dataset_name, "graph_params": gp}
        _write_manifest(index_store, manifest)

        return meta

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No index built for '{dataset_id}'. Call POST /api/datasets/{{id}}/index first."
            )
        return entry

    # ------------------------------------------------------------------
    # Helpers shared by search methods
    # ------------------------------------------------------------------

    def _vec(self, query: dict[str, Any]) -> np.ndarray:
        """Extract and validate 'vector' from query dict, return float64 array."""
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
                [{"index": int(i), "score": float(s)} for i, s in hits] for hits in batch_hits
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
        return {
            "method": "spot_motives_eigen",
            "results": self._spot_hits(entry.aspace.spot_motives_eigen()),
        }

    def spot_motives_energy(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {
            "method": "spot_motives_energy",
            "results": self._spot_hits(entry.aspace.spot_motives_energy()),
        }

    def spot_subg_centroids(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {
            "method": "spot_subg_centroids",
            "results": self._spot_hits(entry.aspace.spot_subg_centroids()),
        }

    def spot_subg_motives(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        return {
            "method": "spot_subg_motives",
            "results": self._spot_hits(entry.aspace.spot_subg_motives()),
        }

    def graph_export(self, dataset_id: str, fmt: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        if fmt == "csr":
            csr_data, csr_indices, csr_indptr, csr_shape = entry.gl.to_csr()
            return {
                "dataset_id": dataset_id,
                "fmt": "csr",
                "data": [float(v) for v in csr_data],
                "indices": [int(v) for v in csr_indices],
                "indptr": [int(v) for v in csr_indptr],
                "shape": list(csr_shape),
            }
        elif fmt == "dense":
            dense = entry.gl.to_dense()
            return {
                "dataset_id": dataset_id,
                "fmt": "dense",
                "matrix": dense.tolist(),
                "nnodes": int(dense.shape[0]),
                "shape": list(dense.shape),
            }
        else:
            raise ValueError(f"fmt must be 'csr' or 'dense', got '{fmt}'")

    def spectral_metrics(self, dataset_id: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        lam_arr = np.array(list(entry.aspace.lambdas()), dtype=np.float64)
        lam_sorted_raw = entry.aspace.lambdas_sorted()
        lam_sorted = [[float(v), int(i)] for v, i in lam_sorted_raw]

        n = len(lam_arr)
        lam_sorted_vals = sorted(lam_arr)

        eps = 1e-10
        nonzero = [v for v in lam_sorted_vals if v > eps]
        fiedler = float(nonzero[0]) if nonzero else 0.0
        spectral_gap = float(nonzero[1] - nonzero[0]) if len(nonzero) >= 2 else 0.0

        lam_max = float(lam_arr.max()) if n > 0 else 1.0
        spectral_energy_total = float(np.sum(lam_arr**2) / n) if n > 0 else 0.0
        spectral_energy_norm = (spectral_energy_total / (lam_max**2)) if lam_max > eps else 0.0

        percentiles = {}
        if n > 0:
            for p in [10, 25, 50, 75, 90]:
                percentiles[f"p{p}"] = float(np.percentile(lam_arr, p))

        return {
            "dataset_id": dataset_id,
            "nitems": entry.nitems,
            "nclusters": entry.nclusters,
            "lambda_min": float(lam_arr.min()) if n > 0 else 0.0,
            "lambda_max": lam_max,
            "lambda_mean": float(lam_arr.mean()) if n > 0 else 0.0,
            "lambda_std": float(lam_arr.std()) if n > 0 else 0.0,
            "lambda_sum": float(lam_arr.sum()) if n > 0 else 0.0,
            "spectral_gap": spectral_gap,
            "fiedler_value": fiedler,
            "algebraic_connectivity": fiedler,
            "lambdas_sorted": lam_sorted,
            "lambda_percentiles": percentiles,
            "spectral_energy_total": spectral_energy_total,
            "spectral_energy_norm": spectral_energy_norm,
        }

    def motives(self, dataset_id: str, mode: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        if mode == "eigen":
            hits = entry.aspace.spot_motives_eigen()
        elif mode == "energy":
            hits = entry.aspace.spot_motives_energy()
        else:
            raise ValueError(f"mode must be 'eigen' or 'energy', got '{mode}'")
        results = [{"index": int(i), "score": float(s)} for i, s in hits]
        return {
            "dataset_id": dataset_id,
            "mode": mode,
            "motives": results,
            "count": len(results),
        }

    def subgraphs(self, dataset_id: str, mode: str) -> dict[str, Any]:
        entry = self._get_entry(dataset_id)
        if mode == "motives":
            hits = entry.aspace.spot_subg_motives()
        elif mode == "centroids":
            hits = entry.aspace.spot_subg_centroids()
        else:
            raise ValueError(f"mode must be 'motives' or 'centroids', got '{mode}'")
        results = [{"index": int(i), "score": float(s)} for i, s in hits]
        return {
            "dataset_id": dataset_id,
            "mode": mode,
            "subgraphs": results,
            "count": len(results),
        }

    def search_with_mode(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        mode = query.get("mode", "taumode")
        dispatch = {
            "taumode": self.search,
            "hybrid": self.search_hybrid,
            "energy": self.search_energy,
            "linear_sorted": self.search_linear_sorted,
        }
        if mode not in dispatch:
            raise ValueError(f"mode must be one of {list(dispatch.keys())}, got '{mode}'")
        result = dispatch[mode](dataset_id, query)
        result["mode"] = mode
        return result

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
    """Return the best available ArrowSpace adapter.

    Priority:
    1. arrowspace package importable  -> _ArrowSpaceAdapter
    2. fallback                       -> _SidecarAdapter

    FIX: broadened except to catch Exception (not just ImportError) because
    the installed arrowspace package raises NameError in __init__.py when its
    internal submodule reference fails.
    """
    from .settings import get_settings

    try:
        import arrowspace as _mod  # type: ignore

        cache_size = get_settings().index_cache_size
        log.info("arrowspace package found; using live adapter (cache_size=%d)", cache_size)
        return _ArrowSpaceAdapter(_mod, cache_size=cache_size)
    except Exception:
        log.info("arrowspace package not available; using sidecar adapter")
        return _SidecarAdapter()


def reset_adapter_cache() -> None:
    if hasattr(load, "cache_clear"):
        load.cache_clear()


# Public alias for the no-op adapter (test compatibility)
_NullAdapter = _UnavailableAdapter
