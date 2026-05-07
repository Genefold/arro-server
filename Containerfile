# Podman-compatible (also a valid Dockerfile). Build with:
#   podman build -t arrospace-server -f Containerfile .
#   docker build -t arrospace-server -f Containerfile .

FROM docker.io/library/python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build prerequisites first to maximise layer reuse.
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --upgrade pip \
    && pip install ".[zarr]"

# Optional: install pyarrow / pyarrowspace at build time by passing build args.
ARG INSTALL_ARROW=0
ARG INSTALL_ARROWSPACE=0
RUN if [ "$INSTALL_ARROW" = "1" ]; then pip install ".[arrow]"; fi \
    && if [ "$INSTALL_ARROWSPACE" = "1" ]; then pip install ".[arrowspace]" || \
        echo "pyarrowspace not installable in this image; sidecar adapter will be used"; fi

EXPOSE 8000
ENV ARROSPACE_HOST=0.0.0.0 \
    ARROSPACE_PORT=8000

CMD ["python", "-m", "arrospace_server"]
