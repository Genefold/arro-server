"""Tests for the Polars path of sidecar_search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from arro_server.arrowspace_adapter import _SidecarAdapter
from arro_server.errors import MetadataUnavailable


@pytest.fixture
def adapter() -> _SidecarAdapter:
    return _SidecarAdapter()


def _write_index(tmp_path: Path, items: list[dict]) -> Path:
    d = tmp_path / "_arrowspace"
    d.mkdir()
    (d / "index.json").write_text(json.dumps({"items": items}))
    return d / "index.json"


class TestPolarsTagsNormalisation:
    def test_tags_as_list(self, tmp_path, adapter):
        index_file = _write_index(tmp_path, [{"id": "a", "tags": ["foo", "bar"]}])
        results = adapter._sidecar_search_polars(index_file, "foo", limit=10)
        assert results[0]["id"] == "a"

    def test_tags_empty_list(self, tmp_path, adapter):
        """Empty arrays inferred as Null should not crash."""
        index_file = _write_index(tmp_path, [{"id": "a", "tags": []}])
        results = adapter._sidecar_search_polars(index_file, "x", limit=10)
        assert results == []

    def test_tags_empty_mixed_with_normal(self, tmp_path, adapter):
        """Mixed empty and non-empty lists."""
        index_file = _write_index(
            tmp_path,
            [
                {"id": "a", "tags": ["foo"]},
                {"id": "b", "tags": []},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "foo", limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "a"


class TestPolarsSearchBehavior:
    def test_search_by_id(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "foo-bar", "tags": ["x"]},
                {"id": "baz", "tags": ["y"]},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "foo", limit=10)
        assert [r["id"] for r in results] == ["foo-bar"]

    def test_search_by_tag(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "a", "tags": ["alpha", "beta"]},
                {"id": "b", "tags": ["gamma"]},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "beta", limit=10)
        assert [r["id"] for r in results] == ["a"]

    def test_search_case_insensitive(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "UPPER-ID", "tags": ["MiXeD"]},
            ],
        )
        assert adapter._sidecar_search_polars(index_file, "upper", limit=10)[0]["id"] == "UPPER-ID"
        assert adapter._sidecar_search_polars(index_file, "mixed", limit=10)[0]["id"] == "UPPER-ID"

    def test_search_limit(self, tmp_path, adapter):
        items = [{"id": f"row-{i}", "tags": ["shared"]} for i in range(10)]
        index_file = _write_index(tmp_path, items)
        results = adapter._sidecar_search_polars(index_file, "shared", limit=3)
        assert len(results) == 3

    def test_search_literal_special_chars(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "a.b", "tags": ["x"]},
                {"id": "axb", "tags": ["y"]},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "a.b", limit=10)
        assert [r["id"] for r in results] == ["a.b"]

    def test_search_returns_tags(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "a", "tags": ["foo", "bar"]},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "a", limit=10)
        assert results[0]["tags"] == ["foo", "bar"]

    def test_tags_null(self, tmp_path, adapter):
        index_file = _write_index(
            tmp_path,
            [
                {"id": "a", "tags": None},
                {"id": "b", "tags": ["foo"]},
            ],
        )
        results = adapter._sidecar_search_polars(index_file, "foo", limit=10)
        assert [r["id"] for r in results] == ["b"]
        results2 = adapter._sidecar_search_polars(index_file, "a", limit=10)
        assert [r["id"] for r in results2] == ["a"]


class TestPolarsPublicAPI:
    def test_sidecar_search_missing_file(self, tmp_path, adapter):
        with pytest.raises(MetadataUnavailable):
            adapter.sidecar_search(tmp_path, "q")

    def test_sidecar_search_ok(self, tmp_path, adapter):
        _write_index(tmp_path, [{"id": "a", "tags": ["foo"]}])
        results = adapter.sidecar_search(tmp_path, "foo", limit=10)
        assert results[0]["id"] == "a"
