from __future__ import annotations

from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Environment variables are prefixed with ``ARRO_SERVER_``.

    ``data_roots`` accepts a comma-separated list, e.g.
    ``ARRO_SERVER_DATA_ROOTS=/data/zarr,/mnt/shared/zarr``.
    Each root may optionally be prefixed with a label: ``label=path``.

    ``cors_origins`` accepts a comma-separated list of allowed origins, e.g.
    ``ARRO_SERVER_CORS_ORIGINS=https://app.example.com,https://admin.example.com``.
    Use ``*`` (the default) to allow all origins — do not use ``*`` in production.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARRO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_roots: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    # default_window: rows returned by /data when ?limit is omitted.
    default_window: int = 100
    # max_window: hard cap on the number of *rows* (leading-axis elements)
    # returned in a single /data or /slice response. Note that for N-D arrays
    # the total element count is max_window * product(shape[1:]).
    max_window: int = 10_000
    serve_frontend: bool = True
    frontend_dir: str | None = None
    # Directory where graph-Laplacian Zarr arrays are persisted.
    index_store: str = "./arrowspace_index"
    # Maximum number of ArrowSpace indices to keep in memory simultaneously.
    # Oldest entry is evicted when the limit is reached.
    index_cache_size: int = 8
    # Optional static bearer token protecting /api/admin/* endpoints.
    # Leave unset (default None) for unauthenticated access (dev/private-network only).
    admin_token: str | None = None

    @field_validator("data_roots", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @cached_property
    def resolved_roots(self) -> dict[str, Path]:  # type: ignore[override]
        """Return mapping of root label -> filesystem path.

        Roots may be specified as ``path`` or ``label=path``.
        Unlabeled roots are auto-named after their directory basename, with
        numeric suffixes on collision.

        Cached as a ``cached_property`` — resolved only once per Settings
        instance, avoiding repeated ``Path.resolve()`` syscalls per request.
        """
        out: dict[str, Path] = {}
        for entry in self.data_roots:
            if "=" in entry:
                label, raw = entry.split("=", 1)
                label = label.strip()
            else:
                raw = entry
                label = Path(raw).name or "root"
            path = Path(raw).expanduser().resolve()
            base = label
            i = 1
            while label in out:
                i += 1
                label = f"{base}-{i}"
            out[label] = path
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Use ``get_settings.cache_clear()`` in tests to reset between cases.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Test / reload helper — clears the lru_cache on get_settings."""
    get_settings.cache_clear()
