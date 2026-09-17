# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install maturin \
    && pip install --prefix=/install --no-deps arrowspace

RUN pip install --prefix=/install \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2.6" \
    "pydantic-settings>=2.10.3" \
    "numpy>=1.26" \
    "zarr>=3.0" \
    "pyarrow>=15.0" \
    "polars>=1.0"

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
    ARRO_SERVER_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "arro_server"]
