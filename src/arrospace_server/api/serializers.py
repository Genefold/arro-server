"""Numpy -> JSON-friendly conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


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
