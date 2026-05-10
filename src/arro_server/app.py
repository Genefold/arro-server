from __future__ import annotations

import importlib.resources
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings


def _configure_logging(settings: Settings) -> None:
    """Set up root logger: JSON lines in production, human-readable in development."""
    if settings.log_json:
        # Structured JSON log lines — one JSON object per line.
        # Compatible with Cloud Logging / Datadog / Loki without a sidecar.
        fmt = (
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "msg": %(message)r}'
        )
    else:
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=fmt,
        force=True,  # override any existing handlers (e.g. uvicorn's defaults)
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    _configure_logging(settings)

    log = logging.getLogger(__name__)
    log.info(
        "Starting arro-server v%s (env=%s, auth=%s, log_json=%s)",
        __version__,
        settings.env,
        bool((settings.api_key or "").strip()),
        settings.log_json,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Pre-load persisted ArrowSpace indices so they survive restarts.
        from .arrowspace_adapter import _ArrowSpaceAdapter
        from .arrowspace_adapter import load as load_arrowspace

        adapter = load_arrowspace()
        if isinstance(adapter, _ArrowSpaceAdapter):
            index_store = settings.effective_index_store()
            n = adapter.load_persisted(index_store)
            if n:
                log.info(
                    "Startup: pre-loaded %d persisted ArrowSpace index(es) from %s",
                    n,
                    index_store,
                )
        yield

    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        allow_credentials=settings.cors_allow_credentials,
    )

    # ------------------------------------------------------------------
    # Rate limiting  (slowapi)
    # ------------------------------------------------------------------
    if settings.rate_limit_write:
        try:
            from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
            from slowapi.errors import RateLimitExceeded  # type: ignore
            from slowapi.util import get_remote_address  # type: ignore

            limiter = Limiter(
                key_func=get_remote_address,
                default_limits=[],  # only apply limits where explicitly declared
            )
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
            log.info("Rate limiting enabled: write endpoints capped at %s", settings.rate_limit_write)
        except ImportError:
            log.warning(
                "slowapi not installed — rate limiting disabled. "
                "Install it with: pip install slowapi"
            )

    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def _root() -> dict[str, str]:
        return {"service": "arro-server", "version": __version__, "docs": "/docs"}

    # ------------------------------------------------------------------
    # Frontend static files
    # ------------------------------------------------------------------
    if settings.serve_frontend:
        frontend_dir: Path | None = None
        if settings.frontend_dir:
            frontend_dir = Path(settings.frontend_dir)
        else:
            # Development layout: <repo>/frontend/
            _dev = Path(__file__).parent.parent.parent / "frontend"
            if _dev.exists():
                frontend_dir = _dev
            else:
                # Installed wheel: share/arro_server/frontend (hatch shared-data)
                try:
                    _pkg = (
                        importlib.resources.files("arro_server")
                        / "../../../share/arro_server/frontend"
                    )
                    _resolved = Path(str(_pkg)).resolve()
                    if _resolved.exists():
                        frontend_dir = _resolved
                except Exception:
                    pass
        if frontend_dir and frontend_dir.exists():
            app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

    return app
