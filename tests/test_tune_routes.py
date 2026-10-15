"""Integration tests for POST and GET /api/datasets/{id}/tune (issue #67)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from arro_server.api.tune_router import _registry, router
from arro_server.deps import get_tuner_adapter
from arro_server.errors import DatasetNotFound
from arro_server.settings import Settings, get_settings
from arro_server.storage.tune_store import TunedParams
from arro_server.tuner_adapter import TunerAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tuned_params(dataset: str = "main--matrix") -> TunedParams:
    return TunedParams(
        eps=0.5,
        k=10,
        topk=3,
        p=2.0,
        sigma=1.0,
        score=0.97,
        tuned_at="2026-09-17T14:00:00+00:00",
        dataset=dataset,
    )


def mock_adapter(
    *,
    running: bool = False,
    params: TunedParams | None = None,
) -> MagicMock:
    """The real TunerAdapter exposes sync is_running/get_params/launch."""
    adapter = MagicMock(spec=TunerAdapter)
    adapter.is_running = MagicMock(return_value=running)
    adapter.get_params = MagicMock(return_value=params)
    adapter.launch = MagicMock(return_value=None)
    return adapter


def mock_registry(nrows: int = 12, ncols: int = 4, exists: bool = True) -> MagicMock:
    """Registry double: open() -> handle with a 2-D array; get_dataset() -> summary."""
    reg = MagicMock(spec=["open", "get_dataset"])
    if not exists:
        reg.open.side_effect = DatasetNotFound("ghost")
        reg.get_dataset.return_value = None
        return reg
    arr = np.arange(nrows * ncols, dtype="float32").reshape(nrows, ncols)
    handle = MagicMock()
    handle.summary.shape = (nrows, ncols)
    handle.read_window.return_value = arr
    reg.open.return_value = handle
    reg.get_dataset.return_value = MagicMock(shape=(nrows, ncols))
    return reg


def build_app(adapter: TunerAdapter, reg: MagicMock, settings: Settings) -> FastAPI:
    """Minimal FastAPI app with the tune router and mocked dependencies."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_tuner_adapter] = lambda: adapter
    app.dependency_overrides[_registry] = lambda: reg
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        index_store=str(tmp_path / "index_store"),
        tune_params_path=str(tmp_path / "tune_params.json"),
    )


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/tune
# ---------------------------------------------------------------------------


class TestPostTune:
    async def test_post_returns_202_started_and_exports_npy(self, settings: Settings):
        reg = mock_registry(nrows=12, ncols=4)
        adapter = mock_adapter(running=False)
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/datasets/main--matrix/tune")

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "started"
        assert body["dataset"] == "main--matrix"
        adapter.launch.assert_called_once()
        args, kwargs = adapter.launch.call_args
        assert args[0] == "main--matrix"
        npy_path = Path(args[1])
        assert npy_path.exists()
        np.testing.assert_array_equal(
            np.load(npy_path),
            np.arange(12 * 4, dtype="float32").reshape(12, 4),
        )
        assert kwargs["n_trials"] == 30

    async def test_post_idempotent_returns_running_when_already_in_progress(
        self, settings: Settings
    ):
        reg = mock_registry()
        adapter = mock_adapter(running=True)
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/datasets/main--matrix/tune")

        assert response.status_code == 202
        assert response.json()["status"] == "running"
        adapter.launch.assert_not_called()
        reg.open.assert_not_called()  # no wasted export

    async def test_post_returns_404_for_unknown_dataset(self, settings: Settings):
        reg = mock_registry(exists=False)
        adapter = mock_adapter()
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/datasets/ghost/tune")

        assert response.status_code == 404
        assert "ghost" in response.json()["detail"]
        adapter.launch.assert_not_called()

    async def test_post_body_dataset_mismatch_raises_422(self, settings: Settings):
        reg = mock_registry()
        adapter = mock_adapter()
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/datasets/main--matrix/tune",
                json={"dataset": "other--dataset", "n_trials": 5},
            )

        assert response.status_code == 422
        adapter.launch.assert_not_called()

    async def test_post_forwards_ranges_to_launch(self, settings: Settings):
        reg = mock_registry()
        adapter = mock_adapter()
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            await client.post(
                "/api/datasets/main--matrix/tune",
                json={
                    "dataset": "main--matrix",
                    "eps_range": (0.5, 8.0),
                    "k_range": (4, 32),
                    "n_trials": 11,
                },
            )

        _, kwargs = adapter.launch.call_args
        assert kwargs == {
            "n_trials": 11,
            "eps_low": 0.5,
            "eps_high": 8.0,
            "k_low": 4,
            "k_high": 32,
        }

    async def test_post_rejects_non_2d_dataset_with_422(self, settings: Settings):
        reg = mock_registry()
        reg.open.return_value.summary.shape = (5,)
        adapter = mock_adapter()
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.post("/api/datasets/main--matrix/tune")

        assert response.status_code == 422
        adapter.launch.assert_not_called()


# ---------------------------------------------------------------------------
# Concurrency — guard → export → launch must be atomic per dataset (#80)
# ---------------------------------------------------------------------------


class TestConcurrentPostTune:
    async def test_concurrent_posts_serialize_to_one_launch(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ):
        """Two POSTs while the first is mid-export: second must not launch.

        Deterministic interleave: the first request parks inside the gated
        export while holding the per-dataset lock; the second must suspend on
        the lock and only re-check is_running() after the first has launched.
        """
        from arro_server.api import tune_router

        reg = mock_registry(nrows=12, ncols=4)
        adapter = mock_adapter(running=False)

        def launch(dataset_id, npy_path, **kwargs):
            adapter.is_running.return_value = True

        adapter.launch = MagicMock(side_effect=launch)

        export_started = threading.Event()
        release_export = threading.Event()
        real_export = tune_router._export_dataset_to_npy

        def gated_export(dataset_id, reg_, settings_):
            export_started.set()
            release_export.wait(timeout=5)
            return real_export(dataset_id, reg_, settings_)

        monkeypatch.setattr(tune_router, "_export_dataset_to_npy", gated_export)

        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            t1 = asyncio.create_task(client.post("/api/datasets/race--probe/tune"))
            t2 = asyncio.create_task(client.post("/api/datasets/race--probe/tune"))
            # to_thread keeps the loop free: this resolves only once the first
            # request holds the lock and is parked inside the gated export.
            assert await asyncio.wait_for(asyncio.to_thread(export_started.wait, 5), timeout=10)
            release_export.set()
            r1, r2 = await asyncio.gather(t1, t2)

        adapter.launch.assert_called_once()
        assert sorted(r.json()["status"] for r in (r1, r2)) == ["running", "started"]


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/tune
# ---------------------------------------------------------------------------


class TestGetTuneStatus:
    async def test_get_returns_not_started_before_any_job(self, settings: Settings):
        reg = mock_registry()
        adapter = mock_adapter(running=False, params=None)
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/datasets/main--matrix/tune")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "not_started"
        assert body["params"] is None
        assert body["dataset"] == "main--matrix"

    async def test_get_returns_running_during_job(self, settings: Settings):
        reg = mock_registry()
        adapter = mock_adapter(running=True)
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/datasets/main--matrix/tune")

        body = response.json()
        assert body["status"] == "running"
        assert body["params"] is None

    async def test_get_returns_done_with_params_after_completion(self, settings: Settings):
        reg = mock_registry()
        params = make_tuned_params("main--matrix")
        adapter = mock_adapter(params=params)
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/datasets/main--matrix/tune")

        body = response.json()
        assert body["status"] == "done"
        assert body["params"]["eps"] == 0.5
        assert body["params"]["k"] == 10
        assert body["params"]["topk"] == 3
        assert body["params"]["score"] == 0.97
        assert body["params"]["dataset"] == "main--matrix"

    async def test_get_returns_404_for_unknown_dataset(self, settings: Settings):
        reg = mock_registry(exists=False)
        adapter = mock_adapter()
        async with AsyncClient(
            transport=ASGITransport(app=build_app(adapter, reg, settings)),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/datasets/ghost/tune")

        assert response.status_code == 404
        assert "ghost" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Dependency injection — adapter not registered
# ---------------------------------------------------------------------------


class TestMissingAdapter:
    async def test_missing_adapter_returns_503(self):
        """If lifespan did not register the adapter, routes must 503 (misconfigured)."""
        app = FastAPI()
        app.include_router(router)
        # Intentionally do NOT override get_tuner_adapter
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/datasets/main--matrix/tune")
        assert response.status_code == 503
