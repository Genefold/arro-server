# arro-server

[![Tests](https://github.com/Genefold/arro-server/actions/workflows/ci.yml/badge.svg)](https://github.com/Genefold/arro-server/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Genefold/arro-server)](LICENSE)
[![Zarr v3](https://img.shields.io/badge/zarr-v3-orange?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0zIDNoMTh2MkgzVjN6bTAgNGgxOHYySDF6bTAgNGgxOHYySDN6bTAgNGgxOHYySDN6Ii8+PC9zdmc+)](https://zarr.dev)

FastAPI server that exposes large datasets stored as **Zarr v3** trees and
their **ArrowSpace**-derived metadata over HTTP. Designed as a reusable
boilerplate for ArrowSpace / arrowspace infrastructure: clean adapter
interfaces, optional dependencies, and a tiny vanilla-JS viewer for smoke
testing dataset browsing, slicing, and metadata.

## Quick start
```bash
uv pip install -e .[dev]
uv run src/arro_server/
```

## Layout

```
src/arro_server/   # Python package
  app.py                # FastAPI application factory
  settings.py           # pydantic-settings configuration
  slicing.py            # numpy-style slice/window parsing
  errors.py             # HTTPException subclasses
  arrowspace_adapter.py # pyarrowspace + sidecar JSON adapters
  storage/
    base.py             # StorageBackend protocol
    zarr_fs.py          # filesystem Zarr v3 backend
    registry.py         # multi-backend multiplexer
  api/
    routes.py           # /api/* endpoints
    serializers.py      # ndarray -> JSON helpers
frontend/               # vanilla HTML/CSS/JS viewer
tests/                  # pytest suite
scripts/make_example_data.py
Containerfile
compose.yaml
```

## Endpoints

All routes live under `/api`. Dataset IDs are `<root_label>/<path>`.

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Liveness + optional dep status + configured roots |
| `GET /api/datasets` | List discovered datasets (arrays + groups) |
| `GET /api/datasets/{id}/metadata` | Shape, dtype, chunks, attrs |
| `GET /api/datasets/{id}/data?offset=0&limit=100` | Row-window over leading axis (infinite scroll) |
| `GET /api/datasets/{id}/slice?slice=0:100,:,3` | Numpy-style multi-axis slice |
| `GET /api/datasets/{id}/manifold` | ArrowSpace manifold JSON |
| `GET /api/datasets/{id}/stats` | Basic + ArrowSpace statistics |
| `GET /api/datasets/{id}/search?q=...&limit=20` | Search seam (sidecar or pyarrowspace) |

The slice grammar mirrors numpy: `start:stop:step` per axis, comma-separated,
negative indices supported, omitted axes default to full extent. The current
scaffold returns JSON previews. **Future:** Arrow IPC, Parquet, and NDJSON
streams for large payloads — the response strategy is isolated in
`api/serializers.py` so this evolves without touching routes.

## ArrowSpace integration

`arrowspace_adapter.py` picks the best backend at runtime:

1. **`pyarrowspace`** if importable (preferred).
2. **Sidecar JSON** under `<dataset>/_arrowspace/` (`manifold.json`,
   `stats.json`, `index.json`) — works in environments without the package.

The sidecar layout means a dataset can advertise ArrowSpace metadata even
when the Python package is unavailable, which is what the boilerplate
assumes today.

## Configuration

Environment variables (prefix `arro-server_`, also read from `.env`):

| Var | Default | Notes |
| --- | --- | --- |
| `arro-server_DATA_ROOTS` | `[]` | Comma-separated. Each entry is `path` or `label=path`. |
| `arro-server_CORS_ORIGINS` | `*` | Comma-separated origins. |
| `arro-server_DEFAULT_WINDOW` | `100` | Default `/data` page size. |
| `arro-server_MAX_WINDOW` | `10000` | Hard cap on per-request elements (per leading-axis row). |
| `arro-server_SERVE_FRONTEND` | `true` | Mount `frontend/` at `/ui`. |
| `arro-server_HOST` / `_PORT` / `_RELOAD` | `0.0.0.0` / `8000` / `0` | Uvicorn bind. |

See `.env.example`.

## Local bootstrap

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# generate a sample Zarr root
uv run scripts/make_example_data.py

# point at it and serve
export ARRO_SERVER_DATA_ROOTS="main=$(pwd)/example_data"
uv run src/arro_server
# UI:  http://localhost:8000/ui
# Docs: http://localhost:8000/docs
```

### Pre-commit hook

The repository ships a [pre-commit](https://pre-commit.com/) configuration
that runs **ruff** (linter + formatter) automatically before every commit,
keeping style consistent without manual intervention.

Install the hook once after cloning:

```bash
# pre-commit is included in the dev extras, so no separate install needed
uv run pre-commit install
```

From that point on, every `git commit` will:

1. Run `ruff check --fix` — lint and auto-fix safe violations.
2. Run `ruff format` — enforce consistent formatting.

If either step modifies or rejects a file the commit is aborted; review the
diff, stage the changes, and re-commit.

To run the checks manually across the whole tree without committing:

```bash
uv run pre-commit run --all-files
```

> **Note:** ruff settings (line length, selected rules, per-file ignores) live
> in `[tool.ruff]` inside `pyproject.toml`. Adjust them there if you need to
> loosen or tighten the rules for your fork.

## Tests

```bash
pytest -q
```

Tests cover settings parsing, slice resolution, dataset listing, metadata,
windowed reads, slice reads, manifold/stats/search via sidecar adapter, and
graceful error paths. Tests requiring `zarr` are skipped automatically when
the package is not installed.

## Containers

Both Podman and Docker work — the file is named `Containerfile` (Podman
canonical name) but is a valid Dockerfile.

```bash
# Build
podman build -t arro-server-server -f Containerfile .
# or
docker build -t arro-server-server -f Containerfile .

# Run with a host directory of Zarr data mounted read-only
podman run --rm -p 8000:8000 \
  -v "$(pwd)/example_data:/data:ro,Z" \
  -e arro-server_DATA_ROOTS="main=/data" \
  arro-server-server
```

Compose (works with `docker compose` and `podman-compose`):

```bash
DATA_DIR=$(pwd)/example_data docker compose up --build
# or
DATA_DIR=$(pwd)/example_data podman-compose up --build
```

To bake in `pyarrow` or `pyarrowspace`:

```bash
podman build --build-arg INSTALL_ARROW=1 --build-arg INSTALL_ARROWSPACE=1 \
  -t arro-server-server -f Containerfile .
```

`pyarrowspace` install is best-effort in the build — if the wheel is
unavailable the sidecar adapter is still functional.

## Extending

- **New backends** (S3/GCS, Parquet, Iceberg): implement
  `storage.base.StorageBackend` and register in
  `storage.registry.get_registry()`.
- **Larger payloads**: add Arrow IPC / Parquet response branches in
  `api/serializers.py` and a content-type negotiator in `api/routes.py`.
- **Auth**: add a `Depends` on every route or wrap the router in a
  middleware. The scaffold deliberately stays auth-free.

## Scope and limitations

- Data responses are JSON previews — fine for spreadsheet windowing,
  unsuitable for very wide rows or huge slices. Element budget is enforced
  via `arro-server_MAX_WINDOW`.
- Group datasets are listed but not directly readable through `/data` or
  `/slice`; address their member arrays by ID.
- The sidecar `search` is a naive substring match against `id` and `tags`.
  Real vector / manifold search lands once `pyarrowspace` is wired up.
