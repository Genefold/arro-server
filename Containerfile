# Podman-compatible (also a valid Dockerfile). Build with:
#   podman build -t arro-server -f Containerfile .
#   docker build -t arro-server -f Containerfile .

FROM docker.io/library/python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tools required by arrowspace (Rust extension) and pytrec-eval-terrier (C).
# Must come before any pip install that pulls compiled packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

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
