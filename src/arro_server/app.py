from __future__ import annotations

importlib_import = None
import importlib.resources
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Pre-load persisted ArrowSpace indices so they survive restarts.
        from .arrowspace_adapter import load as load_arrowspace
        from .arrowspace_adapter import _ArrowSpaceAdapter

        adapter = load_arrowspace()
        if isinstance(adapter, _ArrowSpaceAdapter):
            index_store = settings.effective_index_store()
            n = adapter.load_persisted(index_store)
            if n:
                import logging
                logging.getLogger(__name__).info(
                    "Startup: pre-loaded %d persisted ArrowSpace index(es) from %s",
                    n, index_store,
                )
        yield

    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

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

    return app
