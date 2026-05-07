from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="ArroSpace Server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def _root() -> dict[str, str]:
        return {"service": "arrospace-server", "version": __version__, "docs": "/docs"}

    if settings.serve_frontend:
        frontend_dir = (
            Path(settings.frontend_dir)
            if settings.frontend_dir
            else Path(__file__).parent.parent.parent / "frontend"
        )
        if frontend_dir.exists():
            app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

    return app
