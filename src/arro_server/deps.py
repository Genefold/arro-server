"""FastAPI dependency functions for arro-server route handlers."""

from __future__ import annotations

from fastapi import HTTPException, Request

from .tuner_adapter import TunerAdapter


def get_tuner_adapter(request: Request) -> TunerAdapter:
    """Return the TunerAdapter singleton attached to app.state.

    Raises 503 (not 500) if the adapter was never registered - this signals
    a misconfigured server, not a bug in the request itself.
    """
    adapter: TunerAdapter | None = getattr(request.app.state, "tuner_adapter", None)
    if adapter is None:
        raise HTTPException(
            status_code=503,
            detail="TunerAdapter is not available - tuning service not configured.",
        )
    return adapter
