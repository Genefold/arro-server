# E2E Smoke Test Report — arro-server

**Date:** 2026-09-18
**Image:** `arro-server:dev` (built from `main` @ `9340647` + F1/F2 fixes, not yet committed)
**Dataset:** `main--matrix` (1000 x 8, float32 Zarr v3, read-only mounted)
**Server:** `arro-server:0.2.0`, `arrowspace 0.28.1`, backend `arrowspace` (live adapter)
**Goal:** validate the end-to-end path **tune -> index build -> search** as the primary consumption surface for arro-server, and resolve the `linear_sorted` anomaly surfaced in the previous run.

---

## 1. Verdict

**The stack is fit for use.** All smoke phases pass; the two high-severity bugs found in the first run (F1, F2) are fixed, regression-tested at unit/HTTP level, and verified live. The remaining open items are API-ergonomics gaps, not blockers.

| Phase | Check | Result |
|---|---|---|
| 0 | `/api/health`: `arrowspace_available: true` | PASS |
| 1 | `GET /api/datasets` lists `main--matrix` | PASS |
| 2 | `POST /tune` 202 -> poll -> `GET /tune` 200 with params | PASS (was 500 — F1 fixed) |
| 3 | No-body `POST /index` echoes **tuned** params; `nitems` matches rows | PASS (was defaults — F2 fixed) |
| 4 | `taumode` self-query: row 0 first, score 1.0, descending | PASS |
| 5 | Explicit `graph_params` body overrides tuned; search still correct | PASS |
| 6 | `hybrid` self-query: row 0 first | PASS |
| 7 | Health after rebuilds: `indexed_datasets: ["main--matrix"]` | PASS |

---

## 2. Fixed in this cycle (verified live this run)

### F1 — `GET /tune` 500 on valid tuner output (fixed)
`TunedParamsSchema.score` was bounded `[0.0, 1.0]`, but the tuner objective is
`0.7*mrr + 0.2*log1p(fiedler) + 0.1*log1p(var_lambda)` — unbounded above. This run persisted
`score: 2.137` and `GET /api/datasets/main--matrix/tune` returned **200** with the full params object.
Fix: dropped `le=1.0` (`src/arro_server/api/schemas.py`); regression test in `tests/test_tune_schemas.py`.

### F2 — tuned params never reached index build (fixed, three defect sites)
1. `routes._arrowspace()` built a *second* lru-cached adapter with `tune_store=None`.
   Fix: one adapter per app, parked on `app.state.arrowspace_adapter` by the lifespan; the
   dependency returns it (fallback: `load(tune_store)`).
2. The route pre-injected `DEFAULT_GRAPH_PARAMS` when no body was sent, so the adapter's
   `user > tuned > default` chain was unreachable from HTTP. Fix: pass `body.graph_params`
   through unmodified; response echoes the adapter-resolved params.
3. `GET /health` called `load_arrowspace()` bare — `indexed_datasets` came from a third instance.
   Fix: health uses the shared dependency; `admin_reload` now publishes its rebuilt adapter to
   `app.state` instead of discarding it.
Regression tests: `tests/test_tuned_build_wiring.py` (DI wiring + HTTP-level), plus updated
`test_issue_26`, `test_arrowspace`, `test_vectors_stale_signal`, `test_upload` to patch the
dependency seam instead of a module attribute. Full suite: 0 failures (was 6 order-dependent).

---

## 3. Resolved investigation — `linear_sorted` "anomaly" (was flagged last run)

**Finding: not a bug.** `search_linear_sorted` is a **lambda-range lookup over a λτ-sorted index**,
not a semantic linear scan:

- Upstream (`arrowspace-rs`, `search::sorted_index`): items live in a `BTreeMap<OrderedFloat<lambda>>`;
  the mode answers *“which items share the query’s spectral role”*, ranked by |λ − λ_q|, ties broken
  deterministically by id (JOSS/techrxiv preprint §search: O(log N) range queries on λτ bands).
- Empirically confirmed in-container: `linear_sorted`’s returned “score” **is the item’s λτ value**
  (e.g. hit score `0.3460` == `lambdas()[0]`). It is not a similarity in [0, 1].
- The top hit changed between runs (969 -> 835) because the index was rebuilt with different tuned
  params — a different λτ distribution. Consistent with λ-proximity semantics.
- Top hit for the query’s own vector is not guaranteed: many items share similar λτ; self wins only
  if its λ is nearest. That is the mode’s contract, not a defect.

**Mode semantics, as verified this run (self-vector query on `main--matrix`):**

| Mode | Answers | Self rank | Score convention |
|---|---|---|---|
| `taumode` (default) | semantic + spectral NNS | 1st, 1.0 | similarity, descending, [0,1] |
| `hybrid` | cosine + λ-proximity blend (`alpha`) | 1st, 1.0 | similarity, descending |
| `energy` | spectral-energy proximity | 1st | distance-like, ascending, ~0 for self |
| `linear_sorted` | λτ-proximity range lookup | not guaranteed | **the item’s λτ value** |

**Action:** document this in the API surface (OpenAPI descriptions already list the modes without
semantics); do not “fix” `linear_sorted` — nothing to fix in arro-server.

---

## 4. Open items for production use of arro-server (priority order)

1. ~~**Forward `k` in taumode**~~ — **DONE in this cycle.** The adapter now forwards explicit
   `k` where the library supports it (arrowspace>=0.28.1, verified live: `k=3` returns exactly
   3 hits, row 0 first) and returns a clear `501` on older 0.26.x installs instead of a 500.
   `k` omitted keeps the index's `topk`. Tests: `tests/test_search_k_forwarding.py`.
2. ~~**`energy` mode degenerate (~1e-19 scores)**~~ — **RESOLVED as `501`**: the mode reads
   energymaps, produced only by `ArrowSpaceBuilder.build_energy`, which arro-server never
   calls. Both `/search` (mode=energy) and `/search/energy` now return `501` with an
   explanatory message instead of meaningless numbers. Verified live on both routes.
3. **Compose mount is read-only** (`:ro,Z`): the write path (`/upload/init` + `/upload/commit`,
   `vectors/append`) is untestable against the stock compose file. If ingestion via HTTP is part
   of the intended use, provide a compose override profile with a writable data root and extend
   the smoke script with the upload phase (draft exists in the smoke spec).
4. **`taumode` result count is driven by `topk` from graph params** — with (1) landed, an
   explicit `k` is honoured per-request; `k` omitted still returns `topk` hits.
5. **Pre-existing test hygiene** (not blocking): `TestSidecarFallback` fixtures mutate
   `sys.modules` without monkeypatch and fail in certain file-order combinations when the real
   `arrowspace` package is installed; two latent ruff findings in untouched test files.

---

## 5. Reproduction

```bash
DATA_DIR="$PWD/example_data" docker compose up --build -d
BASE=http://localhost:8000
DATASET=main--matrix
curl -sf $BASE/api/health | jq .                        # arrowspace_available must be true
curl -s -X POST $BASE/api/datasets/$DATASET/tune        # 202 {"status":"started"}
# poll GET /tune until status=done
curl -s -X POST $BASE/api/datasets/$DATASET/index       # no body -> tuned params echoed
VEC=$(curl -s "$BASE/api/datasets/$DATASET/data?offset=0&limit=1" | jq -c '.data.rows[0]')
curl -s -X POST $BASE/api/datasets/$DATASET/search \
  -H "Content-Type: application/json" \
  -d "{\"vector\": $VEC, \"mode\": \"taumode\", \"tau\": 1.0}"
# results[0].index == 0, score 1.0
```

Smoke-test summary line for CI: build 200 with tuned params == tune-store params; search
`results[0].index == 0` with `score == 1.0` for the self-vector; health lists the indexed dataset.
