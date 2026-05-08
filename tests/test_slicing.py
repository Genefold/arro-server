from __future__ import annotations

import pytest

from arro_server.slicing import enforce_window_budget, parse_slice


def test_full_slice() -> None:
    rs = parse_slice(":,:", (10, 4))
    assert rs.out_shape == (10, 4)
    assert rs.n_elements == 40


def test_offset_limit() -> None:
    rs = parse_slice(None, (100, 4), offset=10, limit=5)
    assert rs.out_shape == (5, 4)
    assert rs.selectors[0] == slice(10, 15, 1)


def test_collapsed_axis() -> None:
    rs = parse_slice("3,:", (10, 4))
    assert rs.out_shape == (4,)
    assert rs.selectors[0] == 3


def test_negative_index_normalised() -> None:
    rs = parse_slice("-1,:", (10, 4))
    assert rs.selectors[0] == 9


def test_step() -> None:
    rs = parse_slice("0:10:2", (10,))
    assert rs.out_shape == (5,)


def test_invalid_too_many_axes() -> None:
    with pytest.raises(ValueError):
        parse_slice("1,2,3", (10, 4))


def test_invalid_index() -> None:
    with pytest.raises(ValueError):
        parse_slice("99", (10,))


def test_invalid_step_zero() -> None:
    with pytest.raises(ValueError):
        parse_slice("0:5:0", (10,))


def test_budget_enforced() -> None:
    rs = parse_slice(":", (1000,))
    with pytest.raises(ValueError):
        enforce_window_budget(rs, 100)


def test_offset_clamped_past_end() -> None:
    rs = parse_slice(None, (10,), offset=20, limit=5)
    assert rs.out_shape == (0,)
