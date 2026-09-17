from __future__ import annotations

import asyncio
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
from .api.tune_router import router as tune_router
from .errors import MetadataUnavailable, OptionalDependencyMissing
from .settings import Settings, get_settings
from .storage.tune_store import TuneStore
from .tuner_adapter import TunerAdapter

log = logging.getLogger(__name__)


def _make_lifespan(settings: Settings):
    """Return a lifespan context manager closed over *settings*.

    On startup: reloads persisted ArrowSpace indices (non-fatal on failure)
    and registers TuneStore + TunerAdapter on ``app.state``.
    On shutdown: cancels any in-flight tuning tasks.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        from .arrowspace_adapter import load as load_adapter

        tune_path = Path(settings.tune_params_path).expanduser().resolve()
        tune_path.parent.mkdir(parents=True, exist_ok=True)

        tune_store = TuneStore(tune_path)
        tuner_adapter = TunerAdapter(tune_store)
        app.state.tune_store = tune_store
        app.state.tuner_adapter = tuner_adapter
        log.info("[startup] TuneStore initialised at %s", tune_path)

        adapter = load_adapter(tune_store)
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

        running_tasks = list(tuner_adapter._running.values())
        if running_tasks:
            log.info("[shutdown] Cancelling %d in-flight tuning task(s)...", len(running_tasks))
            for task in running_tasks:
                task.cancel()
            await asyncio.gather(*running_tasks, return_exceptions=True)
            log.info("[shutdown] All tuning tasks cancelled.")

    return _lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
        lifespan=_make_lifespan(settings),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    app.include_router(admin_router)
    app.include_router(tune_router)

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
