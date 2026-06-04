from __future__ import annotations

import importlib.resources
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import admin_router
from .api import router as api_router
from .errors import MetadataUnavailable, OptionalDependencyMissing
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Restore persisted ArrowSpace indices on startup.

    Scans ``index_manifest.json`` inside ``settings.index_store`` and
    reloads every previously-built index into the LRU cache via
    ``load_arrowspace()`` (the arrowspace Rust function, not the adapter
    factory).  Safe to call when the manifest is absent or when the
    arrowspace package is not installed — both cases are handled
    gracefully with a log warning and no exception propagation.
    """
    from .arrowspace_adapter import load as load_adapter

    settings = get_settings()
    adapter = load_adapter()
    index_store = Path(settings.index_store).expanduser().resolve()

    try:
        loaded = adapter.reload_from_manifest(index_store)
        if loaded:
            log.info(
                "[startup] Reloaded %d ArrowSpace index(es) from manifest: %s",
                len(loaded),
                loaded,
            )
        else:
            log.info("[startup] No persisted ArrowSpace indices found in %s", index_store)
    except Exception:
        log.warning(
            "[startup] Index reload failed — server starts without pre-loaded indices.",
            exc_info=True,
        )

    yield  # application is now running


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    app.include_router(admin_router)

    @app.exception_handler(MetadataUnavailable)
    async def _metadata_unavailable_handler(
        request: Request, exc: MetadataUnavailable
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(OptionalDependencyMissing)
    async def _optional_dependency_missing_handler(
        request: Request, exc: OptionalDependencyMissing
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def _root() -> dict[str, str]:
        return {"service": "arro-server", "version": __version__, "docs": "/docs"}

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

            @app.get("/ui", include_in_schema=False)
            def _ui_redirect() -> RedirectResponse:
                return RedirectResponse(url="/ui/")

    return app
