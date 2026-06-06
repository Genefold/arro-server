from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_zarr_root(tmp_path: Path) -> Path:
    """Create a Zarr v3 root populated with two arrays + ArrowSpace sidecars.

    Skips the test cleanly if zarr is not installed.
    """
    zarr = pytest.importorskip("zarr")
    root_dir = tmp_path / "datasets"
    root_dir.mkdir()

    # Dataset 1: 2D float (rows x cols)
    ds1_path = root_dir / "matrix"
    arr1 = zarr.open(
        str(ds1_path),
        mode="w",
        shape=(50, 4),
        chunks=(10, 4),
        dtype="float32",
    )
    arr1[:] = np.arange(50 * 4, dtype="float32").reshape(50, 4)
    try:
        arr1.attrs["description"] = "test matrix"
    except Exception:
        pass

    # ArrowSpace sidecars next to dataset 1.
    sidecar = ds1_path / "_arrowspace"
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / "manifold.json").write_text(json.dumps({"dim": 2, "n_points": 50}))
    (sidecar / "stats.json").write_text(json.dumps({"mean": 99.5, "std": 57.7}))
    (sidecar / "index.json").write_text(
        json.dumps(
            {"items": [{"id": "row-0", "tags": ["alpha"]}, {"id": "row-1", "tags": ["beta"]}]}
        )
    )

    # Dataset 2: 1D ints
    ds2_path = root_dir / "vector"
    arr2 = zarr.open(str(ds2_path), mode="w", shape=(20,), chunks=(5,), dtype="int32")
    arr2[:] = np.arange(20, dtype="int32")

    return root_dir


@pytest.fixture
def configured_app(tmp_zarr_root: Path):
    from arro_server import arrowspace_adapter
    from arro_server import settings as settings_mod
    from arro_server.app import create_app
    from arro_server.storage import registry as registry_mod

    os.environ["ARRO_SERVER_DATA_ROOTS"] = f"main={tmp_zarr_root}"
    os.environ["ARRO_SERVER_SERVE_FRONTEND"] = "false"
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()  # full singleton reset for test isolation
    arrowspace_adapter.reset_adapter_cache()
    app = create_app()
    yield app
    os.environ.pop("ARRO_SERVER_DATA_ROOTS", None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND", None)
    settings_mod.reset_settings_cache()
    registry_mod.get_registry.cache_clear()  # full singleton reset for test isolation
    arrowspace_adapter.reset_adapter_cache()
