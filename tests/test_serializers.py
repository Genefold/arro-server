"""Unit tests for serializers.deep_sanitize."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arro_server.api.serializers import deep_sanitize


# ---------------------------------------------------------------------------
# Scalar coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val, expected", [
    (np.uint8(3),       3),
    (np.int32(-7),      -7),
    (np.int64(2**40),   2**40),
    (np.float32(1.5),   1.5),
    (np.float64(2.5),   2.5),
    (np.bool_(True),    True),
    (np.bool_(False),   False),
    (float("nan"),      None),
    (float("inf"),      None),
    (float("-inf"),     None),
    (np.float32("nan"), None),
    (42,                42),      # plain int passes through
    (3.14,              3.14),    # plain float passes through
    ("hello",           "hello"), # str passes through
    (None,              None),    # None passes through
])
def test_deep_sanitize_scalars(val, expected):
    result = deep_sanitize(val)
    if expected is None:
        assert result is None
    else:
        assert result == expected
        assert type(result) is type(expected)


# ---------------------------------------------------------------------------
# Nested structures
# ---------------------------------------------------------------------------

def test_deep_sanitize_dict():
    raw = {
        "count": np.uint8(3),
        "label": "hello",
        "nested": {"v": np.float32(1.5), "flag": np.bool_(True)},
    }
    result = deep_sanitize(raw)
    assert result == {"count": 3, "label": "hello", "nested": {"v": 1.5, "flag": True}}
    assert type(result["count"]) is int
    assert type(result["nested"]["v"]) is float


def test_deep_sanitize_list():
    raw = [np.int32(0), float("nan"), "ok", np.float64(2.0)]
    result = deep_sanitize(raw)
    assert result == [0, None, "ok", 2.0]


def test_deep_sanitize_tuple_preserved():
    raw = (np.uint8(1), np.uint8(2))
    result = deep_sanitize(raw)
    assert result == (1, 2)
    assert isinstance(result, tuple)


def test_deep_sanitize_deeply_nested():
    raw = {"a": [{"b": np.int16(99)}, np.float32("nan")]}
    result = deep_sanitize(raw)
    assert result == {"a": [{"b": 99}, None]}


def test_deep_sanitize_ndarray_leaf():
    raw = {"matrix": np.array([1, 2, 3], dtype=np.uint8)}
    result = deep_sanitize(raw)
    # ndarrays become plain Python lists
    assert result["matrix"] == [1, 2, 3]
    assert isinstance(result["matrix"], list)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_deep_sanitize_already_clean():
    raw = {"x": 1, "y": [2, 3], "z": "text"}
    assert deep_sanitize(raw) == raw
