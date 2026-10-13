"""Regression coverage against the actual ArrowSpace 0.29 bindings."""

import hashlib
from unittest.mock import Mock

import arrowspace
import numpy as np
import pytest
from fastapi import HTTPException

from arro_server.arrowspace_adapter import _ArrowSpaceAdapter


@pytest.fixture
def live_index(tmp_path):
    rows = np.random.default_rng(42).normal(size=(40, 5))
    params = {"eps": 1.0, "k": 6, "topk": 3, "p": 2.0, "sigma": 1.0}
    adapter = _ArrowSpaceAdapter(arrowspace)
    adapter.build_index("test", rows, tmp_path, params)
    return adapter, rows, tmp_path


def test_direct_persistence_does_not_write_to_cwd(live_index, monkeypatch, tmp_path):
    adapter, rows, store = live_index
    monkeypatch.chdir(tmp_path)
    adapter.build_index("second", rows, store)
    entry = adapter._get_entry("second")
    assert entry.aspace.storage_path == str(store)
    assert entry.aspace.dataset_name.startswith("second_")
    assert not (tmp_path / "storage").exists()


def test_reload_and_search_roundtrip(live_index):
    adapter, rows, store = live_index
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.iterdir() if path.is_file()
    }
    before = adapter.search("test", {"vector": rows[0], "tau": 1.0})
    restored = _ArrowSpaceAdapter(arrowspace)
    assert restored.reload_from_manifest(store) == ["test"]
    assert restored.search("test", {"vector": rows[0], "tau": 1.0}) == before
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.iterdir() if path.is_file()
    } == artifacts


def test_single_item_unwraps_features(live_index):
    adapter, rows, _ = live_index
    np.testing.assert_allclose(adapter.get_item("test", 0)["vector"], rows[0])


def test_csr_sidecars_are_persisted(live_index):
    _, _, store = live_index
    for name in ("data.zarr", "indices.zarr", "indptr.zarr", "meta.json"):
        assert (store / "test" / name).exists()


def test_batch_preserves_failed_row_positions(live_index):
    adapter, rows, _ = live_index
    entry = adapter._get_entry("test")
    entry.aspace = Mock(search_batch=Mock(return_value=[[(1, 0.9)], None, [(2, 0.8)]]))
    result = adapter.search_batch("test", {"vectors": rows[:3], "tau": 1.0})
    assert result["results"] == [
        [{"index": 1, "score": 0.9}], None, [{"index": 2, "score": 0.8}],
    ]


def test_motifs_preserve_native_results(live_index):
    adapter, _, _ = live_index
    entry = adapter._get_entry("test")
    expected = entry.aspace.spot_motives_eigen(entry.gl, {})
    assert adapter.spot_motives_eigen("test")["results"] == expected


def test_subgraphs_preserve_native_results(live_index):
    adapter, _, _ = live_index
    entry = adapter._get_entry("test")
    expected = entry.aspace.spot_subg_centroids(entry.gl, {})
    assert adapter.spot_subg_centroids("test")["results"] == expected


def test_reloaded_subgraphs_return_controlled_error(live_index):
    _, _, store = live_index
    restored = _ArrowSpaceAdapter(arrowspace)
    restored.reload_from_manifest(store)
    with pytest.raises(HTTPException) as error:
        restored.spot_subg_centroids("test")
    assert error.value.status_code == 422
    assert "cluster_assignments" in error.value.detail


@pytest.mark.parametrize("method", ["spot_motives_energy", "spot_subg_motives"])
def test_energy_analytics_reject_eigen_index_with_422(live_index, method):
    adapter, _, _ = live_index
    with pytest.raises(HTTPException) as error:
        getattr(adapter, method)("test")
    assert error.value.status_code == 422
    assert "energy mode required" in error.value.detail
