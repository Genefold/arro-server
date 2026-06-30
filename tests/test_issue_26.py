"""Regression tests for issue #26 — async CPU-bound routes.

Issue: sync FastAPI handlers for build_index, admin_reload and delete_index
block a thread-pool thread. For CPU-bound ArrowSpace builds this can saturate
the pool and queue other requests. The fix converts the handlers to async and
offloads blocking adapter work with asyncio.to_thread.
"""

from __future__ import annotations

import threading
import time
from inspect import iscoroutinefunction

import pytest
from fastapi.testclient import TestClient


def _find_route(app, path: str, method: str):
    method = method.upper()
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"Route {method} {path} not found")


@pytest.fixture
def client(configured_app):
    with TestClient(configured_app) as c:
        yield c


def test_build_index_route_is_async(configured_app) -> None:
    """POST /index must be an async handler so the event loop is freed."""
    route = _find_route(configured_app, "/api/datasets/{dataset_id:path}/index", "POST")
    assert iscoroutinefunction(route.endpoint)


def test_delete_index_route_is_async(configured_app) -> None:
    """DELETE /index must be an async handler so the event loop is freed."""
    route = _find_route(configured_app, "/api/datasets/{dataset_id:path}/index", "DELETE")
    assert iscoroutinefunction(route.endpoint)


def test_admin_reload_route_is_async(configured_app) -> None:
    """POST /admin/reload must be an async handler so the event loop is freed."""
    route = _find_route(configured_app, "/api/admin/reload", "POST")
    assert iscoroutinefunction(route.endpoint)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("arrowspace"),
    reason="arrowspace not installed",
)
def test_concurrent_index_builds_complete_without_queueing(
    client: TestClient,
    tmp_zarr_root: object,
    configured_app,
    monkeypatch,
) -> None:
    """Two POST /index calls on different datasets should both succeed promptly.

    Before the fix both handlers run in the FastAPI thread pool; with enough
    concurrent builds the pool saturates and requests queue. After the fix the
    route handlers are async and only the CPU-bound adapter.build_index call is
    offloaded to a thread, keeping the event loop responsive.
    """
    zarr = pytest.importorskip("zarr")
    import numpy as np
    from arro_server.arrowspace_adapter import load as load_arrowspace
    from arro_server.storage import registry as registry_mod

    # Create a second 2-D dataset so we can index two distinct datasets at once.
    ds2_path = tmp_zarr_root / "matrix2"
    arr2 = zarr.open(
        str(ds2_path),
        mode="w",
        shape=(50, 4),
        chunks=(10, 4),
        dtype="float32",
    )
    arr2[:] = np.arange(50 * 4, dtype="float32").reshape(50, 4)

    # Reset the registry singleton so the new dataset is discovered on the
    # next request. The app itself is reused via the configured_app fixture.
    registry_mod.get_registry.cache_clear()

    adapter = load_arrowspace()

    def slow_build(*args, **kwargs):
        # Simulate a CPU-bound build that blocks its worker thread briefly.
        # We return synthetic metadata instead of calling the real adapter so
        # this test exercises route-level concurrency, not the adapter's
        # internal build_and_store thread-safety (which is a separate concern).
        time.sleep(0.2)
        return {"nitems": 50, "nfeatures": 4, "nclusters": 1}

    monkeypatch.setattr(adapter, "build_index", slow_build)

    results: dict[str, object] = {}

    def post_index(dataset_id: str, key: str) -> None:
        results[key] = client.post(f"/api/datasets/{dataset_id}/index")

    t1 = threading.Thread(target=post_index, args=("main--matrix", "first"))
    t2 = threading.Thread(target=post_index, args=("main--matrix2", "second"))

    start = time.monotonic()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.monotonic() - start

    assert results["first"].status_code == 200, results["first"].text
    assert results["second"].status_code == 200, results["second"].text
    # If builds were forced to run serially this would take >= 0.4s.
    # The exact threshold is generous to avoid flakiness on slow CI runners.
    assert elapsed < 0.35, f"Index builds appear serialized; elapsed={elapsed:.2f}s"
