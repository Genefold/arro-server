# Podman-compatible (also a valid Dockerfile). Build with:
#   podman build -t arro-server -f Containerfile .
#   docker build -t arro-server -f Containerfile .

FROM docker.io/library/python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install C build toolchain + Rust (required by arrowspace/maturin and
# transitive deps such as pytrec-eval-terrier that compile native extensions).
# Rust is installed via rustup into /usr/local/rust so it is available to
# all subsequent RUN steps (including the non-root appuser at runtime if
# needed, though maturin only runs at build time).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Install build prerequisites first to maximise layer reuse.
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --upgrade pip \
    && pip install ".[zarr]"

# Optional: install pyarrow at build time.
ARG INSTALL_ARROW=0
# Optional: install arrowspace (pip install arrowspace).
# Repo: https://github.com/tuned-org-uk/pyarrowspace
ARG INSTALL_ARROWSPACE=0
RUN if [ "$INSTALL_ARROW" = "1" ]; then pip install ".[arrow]"; fi \
    && if [ "$INSTALL_ARROWSPACE" = "1" ]; then pip install ".[arrowspace]" || \
        echo "arrowspace not installable in this image; sidecar adapter will be used"; fi

# Run as a non-root user for security.
# Create the index persistence directory before switching user so that
# appuser owns it and _write_manifest / _persist_csr can write to it.
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/arrowspace_index \
    && chown -R appuser:appuser /app/arrowspace_index
USER appuser

EXPOSE 8000
ENV ARRO_SERVER_HOST=0.0.0.0 \
    ARRO_SERVER_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "arro_server"]
