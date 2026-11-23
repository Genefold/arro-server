"""Pin the legacy vector API and on-disk compatibility from before Document DB."""

import sys
from base64 import b64decode
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from shutil import copytree
from tarfile import open as open_tar

import arrowspace
import numpy as np
import pytest
import zarr
from fastapi.testclient import TestClient

from arro_server.app import create_app
from arro_server.arrowspace_adapter import reset_adapter_cache
from arro_server.settings import reset_settings_cache
from arro_server.storage.registry import get_registry

DATASET = "legacy--vectors"
SEARCH = f"/api/datasets/{DATASET}/search"
QUERY = {"vector": [0.98, 0.04, 0.01], "mode": "taumode", "k": 1}


def _configure(monkeypatch, volume: Path) -> None:
    monkeypatch.setenv("ARRO_SERVER_DATA_ROOTS", f"legacy={volume / 'data'}")
    monkeypatch.setenv("ARRO_SERVER_INDEX_STORE", str(volume / "index"))
    monkeypatch.setenv("ARRO_SERVER_TUNE_PARAMS_PATH", str(volume / "tune_params.json"))
    monkeypatch.setenv("ARRO_SERVER_SERVE_FRONTEND", "false")
    reset_settings_cache()
    get_registry.cache_clear()
    reset_adapter_cache()


def _contents(volume: Path) -> dict[Path, str]:
    return {
        path.relative_to(volume): sha256(path.read_bytes()).hexdigest()
        for path in volume.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def legacy_volume(tmp_path, monkeypatch):
    # Some adapter tests remove the native module after installing a fake.
    monkeypatch.setitem(sys.modules, "arrowspace", arrowspace)
    original = tmp_path / "original"
    with open_tar(fileobj=BytesIO(b64decode(LEGACY_ARCHIVE)), mode="r:xz") as archive:
        archive.extractall(original, filter="data")
    assert _contents(original), "missing pre-change volume contents"
    original_bytes = _contents(original)
    candidate = tmp_path / "candidate"
    copytree(original, candidate)
    _configure(monkeypatch, candidate)
    yield candidate
    assert _contents(original) == original_bytes
    get_registry.cache_clear()
    reset_adapter_cache()


def _assert_search(client):
    response = client.post(SEARCH, json=QUERY)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": DATASET,
        "arrowspace_available": True,
        "backend": "arrowspace",
        "mode": "taumode",
        "results": [{"index": 0, "score": pytest.approx(0.999116125337473)}],
    }


def test_legacy_volume_reopens_without_migration(legacy_volume):
    initial_bytes = _contents(legacy_volume)
    with TestClient(create_app()) as client:
        _assert_search(client)
    assert _contents(legacy_volume) == initial_bytes


def test_legacy_upload_build_and_restart(legacy_volume, tmp_path, monkeypatch):
    rows = np.asarray(zarr.open_array(legacy_volume / "data/vectors", mode="r")[:])
    volume = tmp_path / "fresh"
    (volume / "data").mkdir(parents=True)
    _configure(monkeypatch, volume)
    with TestClient(create_app()) as client:
        response = client.post("/api/upload/init", json={"dataset_id": DATASET, "root": "legacy"})
        assert response.status_code == 200, response.text
        upload_path = volume / "data/vectors"
        assert response.json() == {
            "dataset_id": DATASET,
            "root": "legacy",
            "upload_path": str(upload_path.resolve()),
        }
        zarr.save(str(upload_path), rows)
        response = client.post(
            "/api/upload/commit", json={"dataset_id": DATASET, "fs_path": str(upload_path)}
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "dataset_id": DATASET,
            "registered": True,
            "shape": [6, 3],
            "dtype": "float64",
            "chunks": [6, 3],
            "index_stale": False,
        }
        graph_params = {"eps": 1.31, "k": 2, "topk": 1, "p": 1.8, "sigma": 0.535}
        response = client.post(
            f"/api/datasets/{DATASET}/index", json={"graph_params": graph_params}
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": DATASET,
            "built": True,
            "graph_params": graph_params,
            "nitems": 6,
            "nfeatures": 3,
            "nclusters": 2,
        }
        _assert_search(client)

    initial_bytes = _contents(volume)
    _configure(monkeypatch, volume)
    with TestClient(create_app()) as client:
        _assert_search(client)
    assert _contents(volume) == initial_bytes


# Base64 tar.xz captured from 6a9d4ac (ArrowSpace 0.29.0), via the same
# upload and explicit build endpoints above. Synthetic rows only; storing
# the bytes avoids regenerating the legacy format with candidate code.
LEGACY_ARCHIVE = (
    "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"
    "AAAEWVo="
)
