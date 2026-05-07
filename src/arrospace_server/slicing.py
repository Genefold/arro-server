"""Parse user-supplied slice/window specifications.

We accept two complementary formats, both URL-friendly:

- ``slice`` query parameter (string): numpy-style, comma-separated per axis.
  Each axis spec is one of:
    - ``""`` or ``":"``  -> full axis
    - ``"i"``            -> single index (axis collapses)
    - ``"start:stop"``   -> half-open range
    - ``"start:stop:step"``
  Negative indices and omitted bounds are supported, mirroring numpy.
  Example: ``slice=0:100,:,3``

- ``offset`` + ``limit`` (ints): convenience for the leading axis, suitable
  for infinite scrolling. Other axes default to full extent.

The output is a normalized tuple of Python ``slice`` / ``int`` objects sized
to the dataset rank, plus the resulting shape (with collapsed axes removed).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSlice:
    selectors: tuple[slice | int, ...]
    out_shape: tuple[int, ...]
    n_elements: int


def _parse_axis(spec: str, length: int) -> slice | int:
    spec = spec.strip()
    if spec == "" or spec == ":":
        return slice(0, length, 1)
    if ":" not in spec:
        try:
            i = int(spec)
        except ValueError as e:
            raise ValueError(f"axis index must be int, got {spec!r}") from e
        if i < 0:
            i += length
        if not 0 <= i < length:
            raise ValueError(f"index {spec} out of bounds for axis length {length}")
        return i
    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' in axis spec {spec!r}")
    while len(parts) < 3:
        parts.append("")
    start_s, stop_s, step_s = parts
    step = int(step_s) if step_s else 1
    if step == 0:
        raise ValueError("slice step cannot be zero")
    if start_s == "":
        start = 0 if step > 0 else length - 1
    else:
        start = int(start_s)
        if start < 0:
            start += length
        start = max(0, min(start, length))
    if stop_s == "":
        stop = length if step > 0 else -1
    else:
        stop = int(stop_s)
        if stop < 0:
            stop += length
        if step > 0:
            stop = max(0, min(stop, length))
        else:
            stop = max(-1, min(stop, length - 1))
    return slice(start, stop, step)


def _slice_length(s: slice) -> int:
    start, stop, step = s.start, s.stop, s.step
    if step > 0:
        return max(0, (stop - start + step - 1) // step)
    return max(0, (start - stop + (-step) - 1) // (-step))


def parse_slice(
    spec: str | None,
    shape: tuple[int, ...],
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> ResolvedSlice:
    if not shape:
        raise ValueError("cannot slice 0-d array")
    rank = len(shape)
    if spec:
        axis_specs = spec.split(",")
        if len(axis_specs) > rank:
            raise ValueError(f"slice has {len(axis_specs)} axes, dataset has {rank}")
        while len(axis_specs) < rank:
            axis_specs.append("")
        selectors = tuple(_parse_axis(a, shape[i]) for i, a in enumerate(axis_specs))
    else:
        # offset/limit convenience: applies to leading axis only.
        sels: list[slice | int] = [slice(0, n, 1) for n in shape]
        if offset is not None or limit is not None:
            o = offset or 0
            if o < 0:
                o = max(0, shape[0] + o)
            o = min(o, shape[0])
            stop = shape[0] if limit is None else min(shape[0], o + max(0, limit))
            sels[0] = slice(o, stop, 1)
        selectors = tuple(sels)

    out_dims: list[int] = []
    for sel in selectors:
        if isinstance(sel, slice):
            out_dims.append(_slice_length(sel))
    out_shape = tuple(out_dims)
    n = 1
    for d in out_shape:
        n *= d
    return ResolvedSlice(selectors=selectors, out_shape=out_shape, n_elements=n)


def enforce_window_budget(rs: ResolvedSlice, max_elements: int) -> None:
    if rs.n_elements > max_elements:
        raise ValueError(
            f"requested window has {rs.n_elements} elements, exceeds max {max_elements}"
        )
