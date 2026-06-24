# syntax=docker/dockerfile:1
# ── Stage 1: builder — compila arrowspace da Rust ────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Toolchain Rust necessaria solo in questo stage
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --upgrade pip \
    && pip install . --prefix=/install

# ── Stage 2: runtime — solo il necessario ────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copia solo i binari compilati — niente gcc/g++/Rust
COPY --from=builder /install /usr/local
COPY --from=builder /app/src ./src
COPY --from=builder /app/frontend ./frontend

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
