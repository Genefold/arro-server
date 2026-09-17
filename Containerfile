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

# Copy only what pip needs to resolve deps — avoids invalidating this layer on src changes
COPY pyproject.toml README.md ./

# Single source of truth: install exactly what [project].dependencies declares.
# arrowspace is excluded here so its (expensive, Rust) sdist build stays cached in its own layer.
RUN pip install --prefix=/install $(python -c "import tomllib; deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; print(' '.join(d for d in deps if not d.startswith('arrowspace')))")

# arrowspace ships no linux wheel; built from sdist with full dep resolution, which
# pulls its own requirements (numpy, pyarrow, pandas, scikit-learn) automatically.
# Specifier is read from pyproject.toml so a constraint change alone invalidates this layer.
RUN pip install --prefix=/install \
    $(python -c "import tomllib; deps = tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; print(next(d for d in deps if d.startswith('arrowspace')))")

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
