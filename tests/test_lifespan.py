"""Tests for TuneStore/TunerAdapter registration in the FastAPI lifespan."""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arro_server.app import _make_lifespan, create_app
from arro_server.settings import Settings
from arro_server.storage.tune_store import TuneStore
from arro_server.tuner_adapter import TunerAdapter


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        index_store=str(tmp_path / "index_store"),
        tune_params_path=str(tmp_path / "tune_params.json"),
    )


@contextmanager
def _lifespan_client(settings: Settings) -> Iterator[TestClient]:
    """TestClient with arrowspace_adapter.load patched out (no Rust extension needed)."""
    with patch("arro_server.arrowspace_adapter.load") as mock_load:
        mock_load.return_value.reload_from_manifest.return_value = []
        with TestClient(create_app(settings)) as client:
            yield client


# -- app.state registration ---------------------------------------------------


class TestLifespanRegistration:
    def test_tune_store_attached_to_app_state(self, test_settings):
        with _lifespan_client(test_settings) as client:
            assert isinstance(client.app.state.tune_store, TuneStore)

    def test_tuner_adapter_attached_to_app_state(self, test_settings):
        with _lifespan_client(test_settings) as client:
            assert isinstance(client.app.state.tuner_adapter, TunerAdapter)

    def test_tuner_adapter_uses_correct_tune_store(self, test_settings):
        with _lifespan_client(test_settings) as client:
            assert client.app.state.tuner_adapter._store is client.app.state.tune_store

    def test_tune_params_directory_created(self, tmp_path):
        deep_path = tmp_path / "deep" / "nested" / "tune.json"
        settings = Settings(
            index_store=str(tmp_path / "index"),
            tune_params_path=str(deep_path),
        )
        with _lifespan_client(settings):
            assert deep_path.parent.exists()


# -- Settings threading ---------------------------------------------------------


class TestSettingsThreading:
    def test_injected_settings_used_in_lifespan(self, tmp_path):
        """create_app(settings) must not fall back to get_settings() in lifespan."""
        custom_path = tmp_path / "custom" / "tune.json"
        settings = Settings(
            index_store=str(tmp_path / "index"),
            tune_params_path=str(custom_path),
        )
        with _lifespan_client(settings) as client:
            assert client.app.state.tune_store._path == custom_path.resolve()


# -- Shutdown cancellation --------------------------------------------------------


class TestShutdownCancellation:
    def test_in_flight_tasks_cancelled_on_shutdown(self, test_settings):
        """Inject a never-finishing task into _running; lifespan exit must cancel it."""

        async def _drive() -> None:
            app = FastAPI()
            cm = _make_lifespan(test_settings)(app)
            await cm.__aenter__()
            task = asyncio.get_running_loop().create_task(asyncio.sleep(9999))
            app.state.tuner_adapter._running["fake_dataset"] = task
            await cm.__aexit__(None, None, None)
            assert task.cancelled()

        with patch("arro_server.arrowspace_adapter.load") as mock_load:
            mock_load.return_value.reload_from_manifest.return_value = []
            asyncio.run(_drive())

    def test_shutdown_with_no_running_tasks_is_safe(self, test_settings):
        with _lifespan_client(test_settings) as client:
            assert client.app.state.tuner_adapter._running == {}


# -- Index reload failure is non-fatal ----------------------------------------------


class TestIndexReloadFaultTolerance:
    def test_index_reload_failure_does_not_prevent_tune_store_init(self, test_settings):
        with patch("arro_server.arrowspace_adapter.load") as mock_load:
            mock_load.return_value.reload_from_manifest.side_effect = RuntimeError("disk error")
            app = create_app(test_settings)
            with TestClient(app):
                assert isinstance(app.state.tune_store, TuneStore)
