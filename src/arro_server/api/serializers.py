"""Numpy -> JSON-friendly conversion helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_scalar(v: Any) -> Any:
    """Coerce a scalar to a JSON-safe Python type.

    Handles numpy scalars, float NaN/Inf (replaced with None), and
    complex numbers.  Plain Python types pass through unchanged.
    """
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def deep_sanitize(obj: Any) -> Any:
    """Recursively coerce all numpy scalars / non-finite floats in *obj*.

    Walks dicts, lists, and tuples; applies :func:`_safe_scalar` to every
    leaf value.  This must be called on any dict that originates from Zarr
    ``arr.attrs`` before it is returned to FastAPI, because Pydantic's JSON
    serialiser cannot handle numpy scalar types such as ``numpy.uint8``.

    Examples::

        deep_sanitize({"count": np.uint8(3), "nested": {"v": np.float32(1.5)}})
        # -> {"count": 3, "nested": {"v": 1.5}}

        deep_sanitize([np.int32(0), float("nan"), "ok"])
        # -> [0, None, "ok"]
    """
    if isinstance(obj, dict):
        return {k: deep_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [deep_sanitize(v) for v in obj]
        return sanitized if isinstance(obj, list) else tuple(sanitized)
    return _safe_scalar(obj)


def array_to_payload(arr: np.ndarray, *, preview_max_rows: int | None = None) -> dict[str, Any]:
    """Convert an ndarray to a JSON-friendly preview payload.

    For 2-D arrays we emit a row-oriented ``rows`` field (truncated to
    ``preview_max_rows`` if provided). Otherwise we emit a flat ``values``
    list along with ``shape`` so the client can reshape.
    """
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.dtype.kind in {"S", "U", "O"}:
        # Stringy or object arrays: convert via tolist().
        if arr.ndim == 2:
            rows = arr.tolist()
            if preview_max_rows is not None:
                rows = rows[:preview_max_rows]
            payload["rows"] = rows
        else:
            payload["values"] = arr.tolist()
        return payload

    if arr.dtype.kind in {"c"}:
        # Complex -> {real, imag} pairs.
        flat = arr.reshape(-1)
        payload["values"] = [{"re": float(x.real), "im": float(x.imag)} for x in flat]
        return payload

    if arr.ndim == 2:
        rows = arr.tolist()
        if preview_max_rows is not None:
            rows = rows[:preview_max_rows]
        payload["rows"] = rows
    elif arr.ndim == 1:
        payload["values"] = arr.tolist()
    else:
        payload["values"] = arr.reshape(-1).tolist()
    return payload
