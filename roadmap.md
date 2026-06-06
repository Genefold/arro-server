#25 (thread-safety)     ← prima, è un fix di correttezza --> PR aperta
#27 (register_dataset)  ← prerequisito strutturale
#21 (POST /upload)      ← usa register_dataset dal giorno 1
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

