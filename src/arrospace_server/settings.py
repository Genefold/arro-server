from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Environment variables are prefixed with ``ARROSPACE_``.

    ``data_roots`` accepts a comma-separated list, e.g.
    ``ARROSPACE_DATA_ROOTS=/data/zarr,/mnt/shared/zarr``.
    Each root may optionally be prefixed with a label: ``label=path``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARROSPACE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_roots: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    default_window: int = 100
    max_window: int = 10_000
    serve_frontend: bool = True
    frontend_dir: str | None = None

    @field_validator("data_roots", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    def resolved_roots(self) -> dict[str, Path]:
        """Return mapping of root label -> filesystem path.

        Roots may be specified as ``path`` or ``label=path``.
        Unlabeled roots are auto-named after their directory basename, with
        numeric suffixes on collision.
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


_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reset_settings_cache() -> None:
    """Test helper."""
    global _cached
    _cached = None
