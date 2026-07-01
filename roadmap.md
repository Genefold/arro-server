#25 (thread-safety)     ← fix di correttezza --> PR #28 merged
#27 (register_dataset)  ← prerequisito strutturale --> PR #31 merged
#21 (POST /upload)      ← usa register_dataset dal giorno 1 --> PR #32 merged
#22 (DELETE /datasets)  ← parallelizzabile con #21
#26 (async build)       ← dopo che #21 è stabile


## fix(#25): thread-safe LRUIndexCache + atomic manifest writes #28

## Summary

Thread-safety fixes for  and manifest writes in .

### Changes

1. **** — replaced bare  with -protected access on all five public methods ()
2. **** — added module-level  that wraps the read→modify→write cycle on  in both  and 
3. **** — manifest write is now atomic;  resolved inside the lock for consistency
4. **** — manifest read/pop/write wrapped in the lock; Parquet/CSR cleanup runs outside the lock
5. **Tests** —  with 9 concurrency regression tests covering cache and manifest races

### Closes

Closes #25

are you sure that Python threads is the best abstraction for this? what are the criteria used to choose among different concurrency paradigms? how this extends to read/write data files instead of manifest?

Great questions — answering all three in order.

---

## 1. Are Python threads the best abstraction here?

Yes, for this specific runtime model. arro-server runs as a **single Uvicorn worker**: one CPython process, one asyncio event loop, plus a thread pool that FastAPI uses to run synchronous (`def`) route handlers. In this model the three available concurrency primitives map as follows:

| Primitive | When correct | Here |
|---|---|---|
| `threading.Lock` | Shared in-process state accessed from OS threads | ✅ Exactly this: cache + manifest accessed from sync handlers in FastAPI's thread pool |
| `asyncio.Lock` | Shared state accessed from coroutines on the same event loop | ❌ `build_index` is synchronous — an `asyncio.Lock` held outside the event loop protects nothing |
| `multiprocessing.Lock` | Shared state across separate OS processes | ❌ Single worker, single process — unnecessary overhead |

The choice of `threading.RLock` / `threading.Lock` is not a tradeoff — it is the only correct answer for the current deployment model.

**Does it survive issue #26 (async build_index)?** Yes. When `build_index` moves to `asyncio.to_thread()`, it still runs in an OS thread outside the event loop. `threading.Lock` continues to protect it correctly. An `asyncio.Lock` would *not* protect code running inside `to_thread`.

---

## 2. Criteria for choosing a concurrency paradigm

Three axes:

**Axis 1 — Where does the shared state live?**
- In-process, thread pool → `threading.Lock`
- In-process, event loop → `asyncio.Lock`
- Multi-process, same host → `multiprocessing.Lock` or `filelock.FileLock`
- Multi-host → distributed lock (Redis `SET NX PX`, etcd)

**Axis 2 — Who executes the critical section?**
- Synchronous Python code or blocking calls → `threading.Lock`
- `async def` coroutines → `asyncio.Lock`
- Synchronous code inside `asyncio.to_thread` → `threading.Lock` (runs in an OS thread, outside the event loop)

**Axis 3 — How long is the critical section held?**
- Microseconds (dict ops) → any lock, overhead negligible
- Milliseconds (file I/O) → `threading.Lock` is fine; the manifest lock is held for ~1 ms (read JSON + modify + write JSON)
- Seconds (CPU-bound build, network) → do not hold a lock at all; use a task queue (ARQ/Celery — already documented in the `TODO(multi-worker)` comment in issue #26)

---

## 3. How does this extend to read/write of data files?

The manifest lock pattern does **not** generalise directly to Zarr/Parquet file I/O. Each case is different:

**`POST /upload` writing a new Zarr array (issue #21)**
No lock needed. `zarr.save()` writes to a unique path per `dataset_id` — two concurrent uploads on different datasets never touch the same path. The only shared state is the registry cache, already protected by `_LRUIndexCache`'s `RLock` via `register_dataset()` (issue #27).

**`DELETE /datasets/{id}` while `POST /search` is reading the same dataset (issue #22)**
This is the real read/write concurrency problem on files. The correct model here is **not** a plain `Lock` (which would serialize all readers) but a **reader-writer lock**: N concurrent readers allowed, exclusive writer during delete. Python stdlib has no `RWLock` — the options are `threading.Condition` or the `readerwriterlock` library. Tracked as a `TODO` to be added when issue #22 is implemented:

```python
# TODO(issue-22): replace threading.Lock with a reader-writer lock
# to allow concurrent search reads during dataset delete.
# Options: threading.Condition (stdlib) or readerwriterlock (lightweight, zero deps)
```

**Multi-replica (already documented in the code)**
`threading.Lock` is invisible to other OS processes. The expansion path already in the `TODO(multi-replica)` comments:
- `filelock.FileLock` — POSIX advisory lock (`fcntl.flock`), works on NFS, single-host multi-process
- Redis `SET NX PX` — multi-host, any deployment topology

No code changes needed now — the `TODO` comments are the design record. The interface of `_LRUIndexCache` and `_MANIFEST_LOCK` does not change when the underlying lock implementation is swapped.


## feat(#27) / fix(#27): O(1) register_dataset + lazy cache on StorageRegistry #31

## Summary

O(1) post-upload dataset registration for `StorageRegistry` (issue #27).

### Changes

1. **`storage/base.py`** — Added `summarize(dataset_id, fs_path)` and `owns_label(label)` to `StorageBackend` Protocol
2. **`storage/zarr_fs.py`** — Implemented `summarize()` and `owns_label()` (reuses existing `_summarize_array`)
3. **`storage/registry.py`** — Added `_cache: dict | None`, `_lock: threading.RLock`, `register_dataset()`, `invalidate()`; refactored `reset_registry_cache()` to call `invalidate()`; rewrote `_backend_for_label()` to use `owns_label()` (no more `hasattr` on private attrs, no silent fallback — explicit `DatasetNotFound`)
4. **`tests/test_registry_register.py`** — 11 tests covering cache correctness, thread safety, Protocol compliance, overwrite semantics, summarize failure integrity, and unknown-label routing

### Closes

Closes #27


## feat(#21): POST /upload/init + POST /upload/commit — two-phase upload

## Summary

Two-phase upload API that registers new Zarr datasets into the `StorageRegistry` without triggering an O(N) filesystem rescan. The client calls `/upload/init` to reserve a path, writes the Zarr array to it, then calls `/upload/commit` to insert the dataset into the registry cache via `register_dataset()`.

### API surface

| Endpoint | Request | Response | Behaviour |
|---|---|---|---|
| `POST /api/upload/init` | `{dataset_id, root}` | `{dataset_id, upload_path, root}` | Validates root + dataset_id label, returns absolute filesystem path. **Does not write to disk.** |
| `POST /api/upload/commit` | `{dataset_id, fs_path}` | `{dataset_id, shape, dtype, chunks, index_stale}` | Path-traversal guard → `register_dataset()` → `open()` → response |

### Design choices

**1. Two-phase protocol, not a single POST with a file body.**
Zarr arrays are directories (or S3 prefixes), not single files. Accepting a raw file body would require the server to reassemble a directory tree — complex, fragile, and incompatible with the Zarr v3 spec. Instead the server returns a path and the client writes the array directly via `zarr.save()` or any Zarr-compatible tool. This is the same model as `POST /upload` in most object stores (presigned URL pattern).

**2. Path-traversal guard (`_assert_path_within_roots`).**
The `fs_path` in the commit body is a client-controlled string. Without validation, a client could pass `../../etc/passwd` or any absolute path. The guard calls `Path.resolve()` on both the candidate path and every configured root, then checks `Path.relative_to()` — this eliminates symlink and `..` traversal tricks. Error messages expose root **labels** only, never filesystem paths (no server layout leakage).

**3. `upload_init` does not create directories or touch files.**
Rationale: if the init handler created the path and the client never committed, the filesystem would accumulate empty directories. Creation is deferred to `zarr.save()` (client-side). The init handler is pure validation + path computation — idempotent and side-effect-free.

**4. `index_stale` flag on commit response.**
When a dataset is overwritten (same `dataset_id` as an existing one with a built ArrowSpace index), the response returns `index_stale: True`. The client should call `POST /datasets/{id}/index` to rebuild. The check happens *before* `register_dataset()` so the stale status reflects the pre-commit state, not the post-commit one.

**5. Validation of Zarr summary after commit (`_validate_zarr_summary`).**
Catches the case where the client called `/upload/commit` before the filesystem flush completed (NFS buffering, partial write). Checks:
- Non-empty shape (scalar arrays are rejected as incomplete)
- No zero dimensions (an array with shape `(0, 32)` is clearly empty)
- Non-empty dtype string
Returns HTTP 422 with a diagnostic message in all three cases.

**6. No `invalidate()` or `list_datasets()` in upload handlers.**
`register_dataset()` inserts into the cache without triggering a rescan. The registry remains O(1) for the upload path. Full rescan is still available via `POST /admin/reload` for externally-written datasets.

### Changes (initial implementation)

1. **`api/schemas.py`** — Four new Pydantic models: `UploadInitRequest`, `UploadInitResponse`, `UploadCommitRequest`, `UploadCommitResponse`.
2. **`api/security.py`** — Created with `assert_path_within_roots()` and `validate_zarr_summary()` (later collapsed into `routes.py`).
3. **`api/routes.py`** — Added `upload_init` and `upload_commit` route handlers; updated endpoint map.
4. **`tests/test_upload.py`** — 17 tests covering init validation, commit happy path, path-traversal guard, error cases, and unit tests for security helpers.

### Refactoring (removal of over-engineering, applied after initial implementation)

**1. `security.py` collapsed into `routes.py` (private helpers).**
The two security functions were only used by `upload_commit`. Extracting them to a separate module added an import cycle boundary with zero callers outside the module. Decision: keep helpers as `_assert_path_within_roots` and `_validate_zarr_summary` in `routes.py`. If a second consumer appears, extract to a shared module.

| Before | After |
|---|---|
| `security.py` — 103 LoC, public API | Deleted |
| `routes.py` — import from `.security` | Inline private helpers |
| Calls: `assert_path_within_roots(...)` | Calls: `_assert_path_within_roots(...)` |

**2. `owns_label()` removed from `StorageBackend` Protocol.**
The `owns_label()` method was added in PR #31 (#27) as an abstract Protocol method. In practice it was a thin wrapper around `label in self._roots` for every backend. This over-engineering was removed:

| Before | After |
|---|---|
| `Protocol` defines `owns_label(label) -> bool` | Removed from Protocol |
| `ZarrFilesystemBackend` implements `owns_label` | Removed |
| `_backend_for_label` iterates + calls `b.owns_label(label)` | Calls `getattr(b, "_roots", None)`, checks `label in roots` |

The new approach is honest about the coupling: the registry knows that filesystem backends expose `_roots`. When a second backend type is added, a public `roots()` property can be added to the Protocol at that point — no speculative abstraction.

### Files touched (final)

```
src/arro_server/api/routes.py               ← +private helpers, +route handlers
src/arro_server/api/schemas.py              ← +4 Pydantic models
src/arro_server/storage/base.py             ← -owns_label() from Protocol
src/arro_server/storage/zarr_fs.py          ← -owns_label() method
src/arro_server/storage/registry.py         ← _backend_for_label uses getattr(_roots)
tests/test_upload.py                        ← NEW: 17 tests
tests/test_registry_register.py             ← removed owns_label mock line
src/arro_server/api/security.py             ← DELETED
```

### Closes

Closes #21


---

## feat(#22): DELETE /api/datasets/{dataset_id} — dataset lifecycle management

### Status
Branch: `feat/22-delete-dataset` | PR: pending

### Problem solved
External services (arro-memory) could not remove datasets over HTTP.
Orphaned Zarr arrays accumulated on disk with no cleanup mechanism.

### API
```
DELETE /api/datasets/{dataset_id:path}
Response: {"id": str, "deleted": true, "index_deleted": bool}
```

### Design decisions

**1. `invalidate_dataset()` instead of `invalidate()` (full cache clear)**
`invalidate()` sets `_cache = None` → O(N) rescan on next `GET /datasets`.
`invalidate_dataset()` removes only the deleted entry → O(1).
Symmetry with `register_dataset()` (O(1) insert) was the deciding factor.
The registry remains hot for all other datasets after a delete.

**2. invalidate-before-rmtree sequence (cache tombstone first)**
Evicting the dataset from cache BEFORE calling `shutil.rmtree` means:
- New requests arriving after eviction will rescan and find either the
  file still present (rmtree in progress) or gone (rmtree complete).
  Both outcomes are correct and consistent.
- This reduces the race window to: requests that already completed
  `reg.open()` and are mid-`read_window()`. See risk #1 below.

**3. HTTP 403 (not 400) for path traversal**
`_assert_path_within_roots` (existing, used by `/upload/commit`) raises 400
because a mismatched path in upload is a malformed request.
`_assert_dataset_path_within_roots` (new, used by DELETE) raises 403
because attempting to delete outside roots is a permissions/security issue.
Both helpers co-exist; the existing one is unchanged for backward compat.

**4. `index_deleted` in response**
`adapter.delete_index()` already returns `bool` (True if index existed).
Surfacing this in the response lets arro-memory know whether to re-index
after a re-upload without making an extra GET /lambdas call.

**5. Route ordering — explicit comment, not guard code**
FastAPI first-match: `DELETE /{id}/index` must be registered before
`DELETE /{id}`. Enforced via an explicit IMPORTANT comment above the
decorator. A runtime guard (e.g. `if dataset_id.endswith("/index"): raise`)
was rejected: it's hacky and doesn't cover all sub-routes. The comment
is the right tool here.

**6. `shutil.rmtree` failure handling**
`FileNotFoundError` → idempotent success (dataset was already gone).
`OSError`/`PermissionError` → HTTP 500 with explicit state description:
  "index deleted, registry evicted, but disk removal failed".
This is honest: we cannot undo steps 3 and 4, but we tell the caller
exactly what state they're in so they can call `POST /admin/reload`
and investigate the filesystem.

### Risks

**Risk 1 — delete-while-reading race [Accepted, documented]**
Requests that completed `reg.open()` before `invalidate_dataset()` and
are mid-`read_window()` will receive `zarr.FileNotFoundError` if `rmtree`
completes during the read. This manifests as HTTP 500 on the concurrent
reader.
- Probability: low (requires two concurrent requests on same dataset_id)
- Impact: HTTP 500 on the reader (not data corruption)
- Mitigation path: per-dataset reader-writer lock (see TODO(issue-22-rwlock))
  When arro-memory becomes multi-client, implement RWLock here.

**Risk 2 — partial rmtree leaves corrupted Zarr on disk**
If `rmtree` fails mid-way (NFS timeout, permission on one chunk file),
the dataset directory exists but is partially deleted. It's out of cache
and its index is deleted.
- At next rescan: `zarr.open()` may fail → dataset not listed (not catastrophic)
- The 500 response + message tells the caller to investigate
- Mitigation: the 500 message explicitly says "call POST /admin/reload
  after manual cleanup"

**Risk 3 — route ordering regression [Mitigated]**
`DELETE /{dataset_id:path}` must come after `DELETE /{dataset_id:path}/index`.
Mitigated by explicit IMPORTANT comment in routes.py. A regression test
(`test_delete_index_route_still_works_after_delete_dataset_added`) was added
to catch this if the order is accidentally changed.

### Future work
- `TODO(issue-22-rwlock)`: per-dataset RWLock for multi-client deployments.
  Integration point: `StorageRegistry.invalidate_dataset()` and
  `ZarrFilesystemBackend.open()`. No interface changes needed — add the
  lock acquisition to existing methods.
- When multi-replica support is added, `invalidate_dataset()` is the
  publication point for a "dataset_deleted" event to remote replicas.
  Interface: `invalidate_dataset(dataset_id)` does not change.

### Files touched
```
src/arro_server/storage/registry.py  ← +invalidate_dataset()
src/arro_server/api/routes.py         ← +_assert_dataset_path_within_roots()
                                         +delete_dataset() route
                                         +endpoint map entry
tests/test_delete_dataset.py          ← NEW: 12 tests
roadmap.md                            ← this entry (not committed)
```

