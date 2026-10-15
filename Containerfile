# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Copy only what pip needs to resolve deps — avoids invalidating this layer on src changes
COPY pyproject.toml README.md ./

# Single source of truth: install exactly what [project].dependencies declares.
# ArrowSpace and the tuner are installed together in their own cached layer below.
RUN pip install --prefix=/install $(python -c "import tomllib; deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; print(' '.join(d for d in deps if not d.startswith('arrowspace')))")

# CPython 3.12 Linux x86_64/aarch64 wheels avoid a Rust build toolchain.
# Resolve both manifest requirements and their runtime dependencies together.
RUN pip install --prefix=/install --only-binary=arrowspace \
    $(python -c "import tomllib; deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; print(' '.join(d for d in deps if d.startswith('arrowspace')))")

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --no-cache-dir --no-deps .

RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /data /app/arrowspace_index \
    && chown appuser:appuser /app /data /app/arrowspace_index

USER appuser

EXPOSE 8000
ENV ARRO_SERVER_HOST=0.0.0.0 \
    ARRO_BIND_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "arro_server"]
