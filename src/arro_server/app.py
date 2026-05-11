from __future__ import annotations

import importlib.resources
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm up the embedder and prompt search engine at startup.

    Both are optional — if the data directory is missing (e.g. running only
    the Zarr/dataset endpoints) the server still starts; endpoints that need
    them will return 503 on first request.
    """
    settings = get_settings()

    # --- warm EmbedderService ---
    try:
        from .embedder import EmbedderService
        EmbedderService.get()  # loads model weights, caches singleton
        log.info("EmbedderService warmed up successfully")
    except Exception as exc:
        log.warning("EmbedderService warm-up skipped: %s", exc)

    # --- warm PromptSearchEngine ---
    try:
        from .search_engine import PromptSearchEngine
        PromptSearchEngine.get()  # builds ArrowSpace index, caches singleton
        log.info("PromptSearchEngine warmed up successfully")
    except Exception as exc:
        log.warning("PromptSearchEngine warm-up skipped: %s", exc)

    yield  # application is now running
    # (shutdown logic here if needed in future)


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
        allow_methods=["GET", "POST"],
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
