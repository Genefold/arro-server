# syntax=docker/dockerfile:1
# Podman-compatible (also a valid Dockerfile). Build with:
#   podman build -t arro-server -f Containerfile .
#   docker build -t arro-server -f Containerfile .
#
# Multistage: builder compiles Rust extension, runtime stage copies only
# the compiled .so and lightweight deps — no gcc/g++/Rust toolchain retained.

# ── Stage 1: builder — compiles arrowspace Rust extension ────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Rust toolchain needed only in this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install maturin first (required to build the arrowspace Rust extension).
# Then install arrowspace WITHOUT its declared dependencies:
#   beir, sentence-transformers, transformers, datasets, tsdae, etc.
# are benchmarking/training deps that are NOT needed at inference runtime.
# We add only the actual runtime deps explicitly below.
RUN pip install --upgrade pip maturin \
    && pip install --prefix=/install --no-deps arrowspace

# Runtime deps of arro-server and arrowspace (no torch, no beir).
RUN pip install --prefix=/install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2.6" \
    "pydantic-settings>=2.10.3" \
    "numpy>=1.26" \
    "zarr>=3.0" \
    "pyarrow>=15.0"

# ── Stage 2: runtime — slim image, no build toolchain ──────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder (no gcc/g++/Rust retained).
COPY --from=builder /install /usr/local

# Copy arro-server source and frontend.
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

# Install arro-server itself (pure Python, no deps resolution needed).
RUN pip install --no-deps .

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /data /app/arrowspace_index \
    && chown appuser:appuser /data /app/arrowspace_index
USER appuser

EXPOSE 8000
ENV ARRO_SERVER_HOST=0.0.0.0 \
    ARRO_SERVER_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "arro_server"]
