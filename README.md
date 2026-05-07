# arro-server

FastAPI server that exposes large datasets stored as **Zarr v3** trees and
their **ArrowSpace**-derived metadata over HTTP. Designed as a reusable
boilerplate for ArrowSpace / arrowspace infrastructure: clean adapter
interfaces, optional dependencies, and a tiny vanilla-JS viewer for smoke
testing dataset browsing, slicing, and metadata.

## Layout

```
src/arro-server_server/   # Python package
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
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# generate a sample Zarr root
python scripts/make_example_data.py

# point at it and serve
export arro-server_DATA_ROOTS="main=$(pwd)/example_data"
python -m arro-server_server
# UI:  http://localhost:8000/ui
# Docs: http://localhost:8000/docs
```

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
