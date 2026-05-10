"""API-key authentication dependency for arro-server.

Behaviour
---------
- If ``ARRO_SERVER_API_KEY`` is **not set** (or empty) the server runs in
  *open mode*: all endpoints are accessible without any credential.  This is
  the default for local development.

- If ``ARRO_SERVER_API_KEY`` is set, every request to a *protected* endpoint
  must supply the matching value in the ``X-API-Key`` request header.
  A missing or wrong key returns HTTP 401 with a plain-text body (no stack
  trace leaks).

Usage
-----
Import ``verify_api_key`` and add it as a FastAPI dependency on any route
you want to protect::

    from ..auth import verify_api_key

    @router.post("/datasets/{dataset_id}/index")
    def build_index(
        dataset_id: str,
        _: None = Depends(verify_api_key),
        ...
    ) -> dict[str, Any]:
        ...

The dependency returns ``None`` on success (caller ignores the value).

Security note
-------------
This is a *static shared secret* suitable for internal / self-hosted
deployments. For public-facing APIs consider replacing this with JWT Bearer
token validation or OAuth2 scopes.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException
from fastapi.security import APIKeyHeader

from .settings import Settings, get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    x_api_key: str | None = Depends(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency that enforces the API-key guard.

    - If no key is configured on the server: always passes (open mode).
    - If a key is configured: header must be present and match exactly.
    """
    configured = (settings.api_key or "").strip()
    if not configured:
        # Open mode — no auth required.
        return
    if not x_api_key or x_api_key != configured:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Supply the correct value in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
