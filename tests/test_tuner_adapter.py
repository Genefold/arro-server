"""Tests for TunerAdapter.

Strategy: mock run_tuning at the module-import level so no real Optuna
study runs.  Use asyncio.run / pytest-asyncio for async tests.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from arro_server.storage.tune_store import TuneStore
from arro_server.tuner_adapter import TunerAdapter, _result_to_tuned_params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_result(status: str = "ok", score: float = 0.95) -> Any:
    """Return a minimal TuneResult-like object."""
    r = MagicMock()
    r.status = status
    r.graph_params = {"eps": 2.5, "k": 20, "topk": 10, "p": 0.9, "sigma": None}
    r.best_score = score
    r.error_code = "some_error" if status != "ok" else None
    r.error_message = "something went wrong" if status != "ok" else None
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> TuneStore:
    return TuneStore(tmp_path / "tune.json")


@pytest.fixture
def adapter(store: TuneStore) -> TunerAdapter:
    return TunerAdapter(store)


@pytest.fixture
def embeddings_file(tmp_path: Path) -> Path:
    """A dummy .npy file that exists on disk (TuneRequest needs a real Path)."""
    p = tmp_path / "vecs.npy"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# _result_to_tuned_params mapping
# ---------------------------------------------------------------------------

class TestResultMapping:
    def test_maps_graph_params_correctly(self):
        result = _fake_result()
        params = _result_to_tuned_params("mnist", result)
        assert params.eps == 2.5
        assert params.k == 20
        assert params.topk == 10
        assert params.sigma is None
        assert params.dataset == "mnist"

    def test_score_falls_back_to_zero_when_none(self):
        result = _fake_result()
        result.best_score = None
        params = _result_to_tuned_params("mnist", result)
        assert params.score == 0.0

    def test_sigma_mapped_when_present(self):
        result = _fake_result()
        result.graph_params["sigma"] = 1.2
        params = _result_to_tuned_params("mnist", result)
        assert params.sigma == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# get_params / is_running before any launch
# ---------------------------------------------------------------------------

class TestAdapterInitialState:
    def test_get_params_returns_none_before_any_launch(self, adapter: TunerAdapter):
        assert adapter.get_params("mnist") is None

    def test_is_running_false_before_launch(self, adapter: TunerAdapter):
        assert adapter.is_running("mnist") is False


# ---------------------------------------------------------------------------
# launch — idempotency
# ---------------------------------------------------------------------------

class TestLaunchIdempotency:
    def test_duplicate_launch_is_noop(
        self, adapter: TunerAdapter, embeddings_file: Path, caplog
    ):
        """A second launch while in-flight must warn and not create a second task."""
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                side_effect=lambda req: _fake_result(),
            ), patch(
                "arro_server.tuner_adapter.asyncio.get_running_loop"
            ) as mock_loop:
                # Simulate the task never completing during the test
                mock_task = MagicMock()
                mock_task.done.return_value = False

                def fake_create_task(coro, **kw):
                    coro.close()  # avoid "never awaited" warning
                    return mock_task

                mock_loop.return_value.create_task.side_effect = fake_create_task

                adapter.launch("mnist", embeddings_file)
                assert adapter.is_running("mnist")

                with caplog.at_level(logging.WARNING, logger="arro_server.tuner_adapter"):
                    adapter.launch("mnist", embeddings_file)

                assert "already in-flight" in caplog.text
                # create_task called only once
                assert mock_loop.return_value.create_task.call_count == 1

        asyncio.run(run())

    def test_second_launch_after_completion_is_allowed(
        self, adapter: TunerAdapter, embeddings_file: Path
    ):
        """After a task finishes, launching again for the same dataset must work."""
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                return_value=_fake_result(),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                # Wait for task to complete
                task = adapter._running.get("mnist")
                if task:
                    await task
                assert not adapter.is_running("mnist")
                # Re-launch must be accepted
                adapter.launch("mnist", embeddings_file, n_trials=1)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# launch — result stored after completion
# ---------------------------------------------------------------------------

class TestResultPersisted:
    def test_result_stored_after_successful_run(
        self, adapter: TunerAdapter, store: TuneStore, embeddings_file: Path
    ):
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                return_value=_fake_result(score=0.88),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                task = adapter._running.get("mnist")
                if task:
                    await task

            params = store.get("mnist")
            assert params is not None
            assert params.score == pytest.approx(0.88)
            assert params.dataset == "mnist"
            assert params.eps == pytest.approx(2.5)

        asyncio.run(run())

    def test_failed_result_not_stored(
        self, adapter: TunerAdapter, store: TuneStore, embeddings_file: Path
    ):
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                return_value=_fake_result(status="tuning_error"),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                task = adapter._running.get("mnist")
                if task:
                    await task

            assert store.get("mnist") is None

        asyncio.run(run())

    def test_exception_from_run_tuning_not_propagated(
        self, adapter: TunerAdapter, store: TuneStore, embeddings_file: Path
    ):
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                side_effect=RuntimeError("optuna exploded"),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                task = adapter._running.get("mnist")
                if task:
                    await task  # must not raise

            assert store.get("mnist") is None

        asyncio.run(run())


# ---------------------------------------------------------------------------
# launch — running state cleaned up on error
# ---------------------------------------------------------------------------

class TestRunningStateCleanup:
    def test_is_running_false_after_error(
        self, adapter: TunerAdapter, embeddings_file: Path
    ):
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                side_effect=RuntimeError("crash"),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                task = adapter._running.get("mnist")
                if task:
                    await task
            assert not adapter.is_running("mnist")

        asyncio.run(run())

    def test_is_running_false_after_success(
        self, adapter: TunerAdapter, embeddings_file: Path
    ):
        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                return_value=_fake_result(),
            ):
                adapter.launch("mnist", embeddings_file, n_trials=1)
                task = adapter._running.get("mnist")
                if task:
                    await task
            assert not adapter.is_running("mnist")

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Concurrent datasets run independently
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_multiple_datasets_run_independently(
        self, adapter: TunerAdapter, tmp_path: Path
    ):
        datasets = ["mnist", "cifar", "imagenet"]
        files = {ds: tmp_path / f"{ds}.npy" for ds in datasets}
        for f in files.values():
            f.write_bytes(b"")

        async def run():
            with patch(
                "arro_server.tuner_adapter.run_tuning",
                side_effect=lambda req: _fake_result(),
            ):
                for ds, path in files.items():
                    adapter.launch(ds, path, n_trials=1)

                tasks = [t for t in adapter._running.values()]
                await asyncio.gather(*tasks, return_exceptions=True)

            for ds in datasets:
                assert adapter.get_params(ds) is not None
                assert not adapter.is_running(ds)

        asyncio.run(run())
