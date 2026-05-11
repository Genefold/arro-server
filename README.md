# arro-server

FastAPI server that exposes large datasets stored as **Zarr v3** trees and
their **ArrowSpace** graph-Laplacian index over HTTP.

Also contains the **LEAF Kaban** semantic search backend: a natural-language
prompt search engine built on [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5),
ArrowSpace spectral indexing, and MMR reranking with salience boosting.

---

## Install

```bash
# API server only (no NL embedding, no notebook deps)
pip install -e .

# + NL search layer (sentence-transformers + torch)
pip install -e '.[nlp]'

# + Jupyter / demo notebook dependencies
pip install -e '.[notebook]'

# + dev / test toolchain
pip install -e '.[dev]'

# Everything at once
pip install -e '.[full]'
```

> **uv users:** replace `pip install` with `uv pip install` throughout.

> **Before building the wheel** (`hatch build` / `pip install -e .`) make sure
> the `frontend/` directory exists — hatchling raises `FileNotFoundError`
> otherwise: `mkdir -p frontend`

---

## Quick start

```bash
# 1. install
uv pip install -e '.[dev]'

# 2. generate sample Zarr data (optional)
uv run scripts/make_example_data.py

# 3. configure
export ARRO_SERVER_DATA_ROOTS="main=$(pwd)/example_data"
export ARRO_SERVER_PROMPT_DATA_DIR="$(pwd)/data/prompt_dataset"

# 4. serve
uv run src/arro_server
# Swagger UI: http://localhost:8000/docs
# Dataset UI: http://localhost:8000/ui
```

---

## Project layout

```
src/arro_server/
  app.py                  # FastAPI application factory + lifespan warmup
  settings.py             # pydantic-settings configuration
  slicing.py              # numpy-style slice/window parsing
  errors.py               # HTTPException subclasses
  arrowspace_adapter.py   # pyarrowspace + sidecar JSON adapters
  search_engine.py        # PromptSearchEngine (ArrowSpace + MMR + salience)
  embedder.py             # EmbedderService (nomic-embed-text-v1.5)
  storage/
    base.py               # StorageBackend protocol
    zarr_fs.py            # filesystem Zarr v3 backend
    registry.py           # multi-backend multiplexer
  api/
    routes.py             # /api/* endpoints
    schemas.py            # Pydantic request/response models
    serializers.py        # ndarray -> JSON helpers
frontend/                 # vanilla HTML/CSS/JS viewer (create dir if missing)
notebooks/                # search_engine_demo.ipynb + tuner results
tests/                    # pytest suite
scripts/make_example_data.py
Containerfile
compose.yaml
```

---

## Endpoints

All routes live under `/api`. Dataset IDs use `<root_label>/<path>`.

### Dataset (Zarr) routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/health` | Liveness + optional dep status + engine readiness |
| `GET` | `/api/datasets` | List discovered datasets |
| `GET` | `/api/datasets/{id}/metadata` | Shape, dtype, chunks, attrs |
| `GET` | `/api/datasets/{id}/data?offset=0&limit=100` | Row-window over leading axis |
| `GET` | `/api/datasets/{id}/slice?spec=0:10,2:5` | NumPy-style multi-axis slice |
| `GET` | `/api/datasets/{id}/manifold` | ArrowSpace manifold JSON |
| `GET` | `/api/datasets/{id}/stats` | Basic + ArrowSpace statistics |
| `GET` | `/api/datasets/{id}/search?q=...` | Keyword search (sidecar or pyarrowspace) |
| `POST` | `/api/datasets/{id}/index` | Build ArrowSpace graph-Laplacian index |
| `GET` | `/api/datasets/{id}/lambdas` | Eigenvalue distribution |
| `GET` | `/api/datasets/{id}/graph_laplacian` | Graph-Laplacian metadata |
| `GET` | `/api/datasets/{id}/items` | All indexed items |
| `GET` | `/api/datasets/{id}/items/{n}` | Single indexed item |
| `POST` | `/api/datasets/{id}/search` | Spectral vector search |
| `POST` | `/api/datasets/{id}/search/energy` | Energy vector search |
| `POST` | `/api/datasets/{id}/search/hybrid` | Hybrid (spectral + linear) search |
| `POST` | `/api/datasets/{id}/search/linear` | Linear sorted search |
| `POST` | `/api/datasets/{id}/search/batch` | Batch spectral search |
| `GET` | `/api/datasets/{id}/spot/motives/eigen` | Eigen motives |
| `GET` | `/api/datasets/{id}/spot/motives/energy` | Energy motives |
| `GET` | `/api/datasets/{id}/spot/subgraphs/centroids` | Subgraph centroids |
| `GET` | `/api/datasets/{id}/spot/subgraphs/motives` | Subgraph motives |

### LEAF Kaban — Prompt search routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/prompts/health` | Engine + embedder readiness |
| `GET` | `/api/prompts/warm` | Confirm index is hot; returns index stats |
| `GET` | `/api/prompts/lambdas` | Eigenvalue distribution for prompt corpus |
| `GET` | `/api/prompts/graph_laplacian` | Graph-Laplacian metadata for prompt corpus |
| `GET` | `/api/prompts/audit` | Degree stats + Fiedler value + PCA 2D scatter |
| `POST` | `/api/prompts/search` | Semantic search — caller supplies 768-d vector |
| `POST` | `/api/prompts/nl_search` | **Primary frontend endpoint** — plain-text query, server embeds it |

#### `POST /api/prompts/nl_search` — example

```bash
curl -X POST http://localhost:8000/api/prompts/nl_search \
  -H 'Content-Type: application/json' \
  -d '{"query": "write a Python function that sorts a list", "k": 5}'
```

```jsonc
{
  "query": "write a Python function that sorts a list",
  "k": 5,
  "tau": 0.75,
  "lam": 0.7,
  "result_count": 5,
  "results": [
    {
      "id": "pk_00042",
      "title": "Python sorting helper",
      "content": "Write a Python function that ...",
      "tags": ["python", "sorting"],
      "score": 0.912,
      "salience": 0.741,
      "tau": 0.75
    }
    // ... 4 more
  ]
}
```

#### `POST /api/prompts/search` — request body

```jsonc
{
  "vector": [0.012, -0.034, /* ... 768 floats total */],
  "k": 10,       // results (1–100, default 10)
  "tau": 0.75,   // spectral sharpness — 0=broad, 5=sharp (default 0.75)
  "alpha": 0.6,  // cosine vs spectral blend (default 0.6)
  "lam": 0.7     // MMR diversity — 1.0=pure relevance, 0.0=max diversity (default 0.7)
}
```

---

## Configuration

All variables are prefixed `ARRO_SERVER_` and can also be set in a `.env` file.

| Variable | Default | Notes |
|----------|---------|-------|
| `ARRO_SERVER_DATA_ROOTS` | `[]` | Comma-separated `path` or `label=path` entries |
| `ARRO_SERVER_PROMPT_DATA_DIR` | — | Directory containing `dataset.json` and `nomic_embs/`. **Required for LEAF Kaban.** |
| `ARRO_SERVER_CORS_ORIGINS` | `*` | Comma-separated allowed origins. Set explicitly in production. |
| `ARRO_SERVER_DEFAULT_WINDOW` | `100` | Default `/data` page size |
| `ARRO_SERVER_MAX_WINDOW` | `10000` | Hard cap on per-request elements |
| `ARRO_SERVER_SERVE_FRONTEND` | `true` | Mount `frontend/` at `/ui` |
| `ARRO_SERVER_HOST` | `0.0.0.0` | Uvicorn bind host |
| `ARRO_SERVER_PORT` | `8000` | Uvicorn bind port |
| `ARRO_SERVER_RELOAD` | `false` | Uvicorn hot-reload (dev only) |

See `.env.example` for a ready-to-copy template.

---

## ArrowSpace integration

`arrowspace_adapter.py` picks the best backend at runtime:

1. **`pyarrowspace`** if importable (preferred).
2. **Sidecar JSON** under `<dataset>/_arrowspace/` (`manifold.json`, `stats.json`, `index.json`) — works without the package.

The sidecar layout means a dataset can advertise ArrowSpace metadata even when the Python package is unavailable.

---

## LEAF Kaban — how it works

```
User NL query
     │
     ▼
EmbedderService          (nomic-embed-text-v1.5 via sentence-transformers)
     │  768-d float64 vector
     ▼
PromptSearchEngine
  ├─ ArrowSpace spectral search   (graph-Laplacian tau-mode)
  │    returns top-3k candidates with cosine scores
  └─ MMR reranker                 (diversity + salience boost)
       salience = f(upvotes, likes, author_reputation, views)
       returns top-k final results
     │
     ▼
PromptSearchResponse     (typed Pydantic, score/salience/tau per result)
```

**Startup warmup:** both `EmbedderService` and `PromptSearchEngine` are
initialised at server startup inside the FastAPI lifespan. If
`ARRO_SERVER_PROMPT_DATA_DIR` is not set the server still boots — prompt
routes return `503` until the data dir is configured and the server restarted.

---

## Pre-commit / linting

```bash
# install the hook once after cloning
uv run pre-commit install

# run manually across the whole tree
uv run pre-commit run --all-files
```

Every `git commit` runs `ruff check --fix` then `ruff format`. Settings live
in `[tool.ruff]` inside `pyproject.toml`.

---

## Tests

```bash
pytest -q
```

Covers settings parsing, slice resolution, dataset listing, metadata,
windowed reads, slice reads, manifold/stats/search via sidecar adapter, and
graceful error paths. Tests requiring `zarr` are skipped automatically when
the package is not installed.

---

## Containers

```bash
# Build
podman build -t arro-server -f Containerfile .
# or
docker build -t arro-server -f Containerfile .

# Run — mount Zarr data + prompt dataset
podman run --rm -p 8000:8000 \
  -v "$(pwd)/example_data:/data:ro,Z" \
  -v "$(pwd)/data/prompt_dataset:/prompts:ro,Z" \
  -e ARRO_SERVER_DATA_ROOTS="main=/data" \
  -e ARRO_SERVER_PROMPT_DATA_DIR="/prompts" \
  arro-server
```

Compose:

```bash
DATA_DIR=$(pwd)/example_data docker compose up --build
```

To bake in `nlp` extras (sentence-transformers + torch):

```bash
podman build --build-arg INSTALL_NLP=1 -t arro-server -f Containerfile .
```

---

## Extending

- **New storage backends** (S3/GCS, Parquet, Iceberg): implement
  `storage.base.StorageBackend` and register in `storage.registry.get_registry()`.
- **Larger payloads**: add Arrow IPC / Parquet response branches in
  `api/serializers.py` and a content-type negotiator in `api/routes.py`.
- **Auth**: add a `Depends` guard on every route or wrap the router in
  middleware. The scaffold is deliberately auth-free.
- **Different embedding model**: swap the model name in `embedder.py` —
  update `_DIM` in `search_engine.py` to match the new output dimension.

---

## Scope and limitations

- Data responses are JSON previews — fine for windowed spreadsheet browsing,
  unsuitable for very wide rows or huge slices. Element budget is enforced
  via `ARRO_SERVER_MAX_WINDOW`.
- Group datasets are listed but not directly readable through `/data` or
  `/slice`; address their member arrays by ID.
- The sidecar keyword search is a naive substring match. Use
  `/api/prompts/nl_search` for semantic search over the prompt corpus.
- NL search requires `pip install 'arro-server[nlp]'` and ~2 GB for the
  torch wheels. A CPU-only install is supported:
  ```bash
  pip install 'arro-server[nlp]' --extra-index-url https://download.pytorch.org/whl/cpu
  ```
