"""Tests for the Polars path of sidecar_search (tags normalisation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pl = pytest.importorskip("polars", reason="polars not installed — skipping Polars path tests")

from arro_server.arrowspace_adapter import _SidecarAdapter


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
        results = adapter._sidecar_search_polars(pl, index_file, "foo", limit=10)
        assert results[0]["id"] == "a"

    def test_tags_empty_list(self, tmp_path, adapter):
        """Empty arrays inferred as Null should not crash."""
        index_file = _write_index(tmp_path, [{"id": "a", "tags": []}])
        results = adapter._sidecar_search_polars(pl, index_file, "x", limit=10)
        assert results == []

    def test_tags_empty_mixed_with_normal(self, tmp_path, adapter):
        """Mixed empty and non-empty lists."""
        index_file = _write_index(tmp_path, [
            {"id": "a", "tags": ["foo"]},
            {"id": "b", "tags": []},
        ])
        results = adapter._sidecar_search_polars(pl, index_file, "foo", limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "a"
