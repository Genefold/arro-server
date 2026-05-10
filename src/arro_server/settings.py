"""arro-server settings.

All configuration is read from environment variables with the ``ARRO_SERVER_``
prefix (or from a ``.env`` file in the working directory).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARRO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Data roots  (label=path pairs, comma-separated)
    # e.g.  ARRO_SERVER_DATA_ROOTS=main=/data/zarr,aux=/data/aux
    # ------------------------------------------------------------------
    data_roots: str = Field(
        default="",
        description="Comma-separated label=path pairs for Zarr dataset roots.",
    )

    # ------------------------------------------------------------------
    # ArrowSpace index store
    # ------------------------------------------------------------------
    arrowspace_index_store: Path = Field(
        default=Path("./storage"),
        description=(
            "Writable directory for persisted ArrowSpace Parquet index files "
            "and the index manifest. "
            "Env var: ARRO_SERVER_ARROWSPACE_INDEX_STORE (preferred) "
            "or ARRO_SERVER_INDEX_STORE (legacy alias)."
        ),
        validation_alias="arrowspace_index_store",
    )

    index_store: Path = Field(
        default=Path("./storage"),
        description="Legacy alias for arrowspace_index_store (ARRO_SERVER_INDEX_STORE).",
    )

    # ------------------------------------------------------------------
    # LRU cache size for in-memory ArrowSpace objects
    # ------------------------------------------------------------------
    index_cache_size: int = Field(
        default=8,
        ge=1,
        description="Maximum number of ArrowSpace indices to keep in memory (LRU).",
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = Field(
        default="*",
        description=(
            "Comma-separated allowed CORS origins. "
            "Use '*' for development only — restrict in production."
        ),
    )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    api_key: str = Field(
        default="",
        description=(
            "Static API key required in the X-API-Key header for write operations "
            "(POST /index, DELETE /index). "
            "Leave empty to run in open mode (no auth). "
            "Env var: ARRO_SERVER_API_KEY."
        ),
    )

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    serve_frontend: bool = Field(
        default=True,
        description="Mount the built-in Vanilla JS UI at /ui.",
    )

    frontend_dir: str = Field(
        default="",
        description="Override path to the frontend static directory (optional).",
    )

    max_window: int = Field(
        default=10_000,
        ge=1,
        description="Hard cap on number of rows returned per data-window request.",
    )

    default_window: int = Field(
        default=1_000,
        ge=1,
        description=(
            "Default number of rows returned by GET /data when 'limit' is not supplied. "
            "Must be <= max_window."
        ),
    )

    # ------------------------------------------------------------------
    # Resolved roots (computed property, not a setting field)
    # ------------------------------------------------------------------

    @property
    def resolved_roots(self) -> dict[str, Path]:
        """Alias kept for backwards compatibility with health endpoint."""
        return self.parsed_data_roots()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _strip_cors(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def parsed_data_roots(self) -> dict[str, Path]:
        """Return {label: Path} from the DATA_ROOTS string."""
        roots: dict[str, Path] = {}
        for part in self.data_roots.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"DATA_ROOTS entry '{part}' must be 'label=path'."
                )
            label, _, path = part.partition("=")
            roots[label.strip()] = Path(path.strip())
        return roots

    def effective_index_store(self) -> Path:
        """Return arrowspace_index_store, falling back to index_store (legacy)."""
        default = Path("./storage")
        if self.arrowspace_index_store != default:
            return self.arrowspace_index_store
        return self.index_store


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()
