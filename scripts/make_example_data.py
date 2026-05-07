"""Create a small Zarr v3 root under ./example_data for smoke testing.

Run after installing the dev extras:

    pip install -e ".[dev]"
    python scripts/make_example_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError as e:  # pragma: no cover
    raise SystemExit("zarr is required: pip install -e '.[dev]'") from e


ROOT = Path(__file__).resolve().parent.parent / "example_data"


def main() -> None:
    ROOT.mkdir(exist_ok=True)

    # Matrix dataset.
    matrix_path = ROOT / "matrix"
    arr = zarr.open(
        str(matrix_path), mode="w", shape=(1000, 8), chunks=(200, 8), dtype="float32"
    )
    rng = np.random.default_rng(42)
    arr[:] = rng.standard_normal(size=(1000, 8)).astype("float32")
    try:
        arr.attrs["description"] = "Synthetic 1000x8 matrix"
    except Exception:  # noqa: BLE001
        pass

    sidecar = matrix_path / "_arrowspace"
    sidecar.mkdir(exist_ok=True)
    (sidecar / "manifold.json").write_text(
        json.dumps(
            {
                "kind": "umap-like",
                "intrinsic_dim": 4,
                "embedding": [[float(x), float(y)] for x, y in rng.standard_normal((50, 2))],
            }
        )
    )
    (sidecar / "stats.json").write_text(
        json.dumps(
            {
                "mean": float(arr[:].mean()),
                "std": float(arr[:].std()),
                "min": float(arr[:].min()),
                "max": float(arr[:].max()),
            }
        )
    )
    (sidecar / "index.json").write_text(
        json.dumps(
            {
                "items": [
                    {"id": f"row-{i}", "tags": ["sample", "synthetic", f"chunk-{i // 200}"]}
                    for i in range(1000)
                ]
            }
        )
    )

    # 3D cube dataset (e.g. images).
    cube_path = ROOT / "cube"
    cube = zarr.open(str(cube_path), mode="w", shape=(20, 32, 32), chunks=(5, 32, 32), dtype="uint8")
    cube[:] = (rng.random(size=(20, 32, 32)) * 255).astype("uint8")

    print(f"wrote example data to {ROOT}")


if __name__ == "__main__":
    main()
