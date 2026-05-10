"""arro-server settings.

All configuration is read from environment variables with the ``ARRO_SERVER_``
prefix (or from a ``.env`` file in the working directory).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARRO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Environment tier
    # ------------------------------------------------------------------
    env: Literal["development", "production"] = Field(
        default="development",
        description=(
            "Deployment tier. Set to 'production' to enable safety guards "
            "(strict CORS, API key required, JSON logging). "
            "Env var: ARRO_SERVER_ENV."
        ),
    )

    # ------------------------------------------------------------------
    # Data roots  (label=path pairs OR bare paths, comma-separated)
    # e.g.  ARRO_SERVER_DATA_ROOTS=main=/data/zarr,aux=/data/aux
    # or    ARRO_SERVER_DATA_ROOTS=/data/zarr,/data/aux  (label auto-derived from stem)
    # ------------------------------------------------------------------
    data_roots: str = Field(
        default="",
        description="Comma-separated label=path pairs (or bare paths) for Zarr dataset roots.",
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

    # Emit structured JSON log lines (one JSON object per line).
    # Automatically set to True when env='production' via the model validator.
    log_json: bool = Field(
        default=False,
        description=(
            "Emit structured JSON log lines. "
            "Defaults to True when ARRO_SERVER_ENV=production. "
            "Env var: ARRO_SERVER_LOG_JSON."
        ),
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = Field(
        default="*",
        description=(
            "Comma-separated allowed CORS origins. "
            "Use '*' for development only. "
            "MUST be restricted (not '*') when ARRO_SERVER_ENV=production."
        ),
    )

    cors_allow_credentials: bool = Field(
        default=False,
        description=(
            "Set to True to allow cookies / Authorization headers from "
            "cross-origin requests. When True, cors_origins must NOT be '*'."
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
            "MUST be set when ARRO_SERVER_ENV=production. "
            "Env var: ARRO_SERVER_API_KEY."
        ),
    )

    # ------------------------------------------------------------------
    # Rate limiting  (uses slowapi / limits library)
    # ------------------------------------------------------------------
    rate_limit_write: str = Field(
        default="30/minute",
        description=(
            "Rate-limit string applied to write endpoints (POST /index, "
            "DELETE /index). Uses the `limits` library syntax: '30/minute', "
            "'5/second', etc. Set to '' to disable rate limiting."
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
    # Validators
    # ------------------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _strip_cors(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _production_safety_checks(self) -> "Settings":
        """Enforce production-safe defaults when ARRO_SERVER_ENV=production.

        Raises ValueError (caught by pydantic as a validation error) if:
        - CORS is still the wildcard '*'
        - API key is not set
        - cors_allow_credentials=True with cors_origins='*'
        """
        if self.cors_allow_credentials and self.cors_origins_list() == ["*"]:
            raise ValueError(
                "cors_allow_credentials=True is incompatible with cors_origins='*'. "
                "Set ARRO_SERVER_CORS_ORIGINS to explicit origin(s)."
            )

        if self.env == "production":
            if self.cors_origins_list() == ["*"]:
                raise ValueError(
                    "ARRO_SERVER_CORS_ORIGINS must not be '*' in production. "
                    "Set it to your frontend origin(s), e.g. "
                    "ARRO_SERVER_CORS_ORIGINS=https://app.example.com"
                )
            if not (self.api_key or "").strip():
                raise ValueError(
                    "ARRO_SERVER_API_KEY must be set in production. "
                    "Generate one with: openssl rand -hex 32"
                )
            # Force JSON logging in production unless explicitly overridden.
            object.__setattr__(self, "log_json", True)

        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def parsed_data_roots(self) -> dict[str, Path]:
        """Return {label: Path} from the DATA_ROOTS string.

        Accepts both:
          - ``label=path`` pairs  (explicit label)
          - bare ``path`` entries (label auto-derived from ``Path(path).stem``)
        """
        roots: dict[str, Path] = {}
        for part in self.data_roots.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                label, _, path = part.partition("=")
                roots[label.strip()] = Path(path.strip())
            else:
                # Bare path — derive label from the final directory component.
                p = Path(part)
                label = p.stem or p.name or part
                roots[label] = p
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
