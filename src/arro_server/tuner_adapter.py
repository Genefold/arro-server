"""TunerAdapter — async non-blocking wrapper around arrowspace_tuner.service.run_tuning.

Design rules:
- run_tuning() is CPU-bound and blocking; it MUST run in the default
  ThreadPoolExecutor via run_in_executor, never directly in the event loop.
- launch() is idempotent: a second call for the same dataset while a job is
  in-flight is a no-op.
- The adapter owns no I/O; TuneStore owns persistence.
- All exceptions from the tuner are caught and logged; they never propagate
  to callers of launch().
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from arrowspace_tuner.models import TuneRequest, TuneResult
from arrowspace_tuner.service import run_tuning

from .storage.tune_store import TunedParams, TuneStore

logger = logging.getLogger(__name__)


def _result_to_tuned_params(dataset: str, result: TuneResult) -> TunedParams:
    """Map a successful TuneResult to the TunedParams storage model."""
    gp = result.graph_params  # guaranteed non-None when status == "ok"
    return TunedParams(
        eps=float(gp["eps"]),
        k=int(gp["k"]),
        topk=int(gp["topk"]),
        p=float(gp["p"]),
        sigma=float(gp["sigma"]) if gp.get("sigma") is not None else None,
        score=float(result.best_score) if result.best_score is not None else 0.0,
        tuned_at=TunedParams.now_utc(),
        dataset=dataset,
    )


class TunerAdapter:
    """Async wrapper around arrowspace_tuner.service.run_tuning.

    Parameters
    ----------
    store:
        A TuneStore instance for persisting results.
    """

    def __init__(self, store: TuneStore) -> None:
        self._store = store
        self._running: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self, dataset: str) -> TunedParams | None:
        """Return last persisted TunedParams for *dataset*, or None."""
        return self._store.get(dataset)

    def is_running(self, dataset: str) -> bool:
        """Return True if a tuning job for *dataset* is currently in-flight."""
        task = self._running.get(dataset)
        return task is not None and not task.done()

    def launch(
        self,
        dataset: str,
        input_path: Path,
        **tuner_kwargs: Any,
    ) -> None:
        """Schedule a tuning job for *dataset* as a background async Task.

        Idempotent: if a job for this dataset is already running, this call
        is a no-op and logs a warning.

        Parameters
        ----------
        dataset:
            Logical name of the dataset; used as the TuneStore key.
        input_path:
            Path to the .npy or .npz embeddings file on disk.
            Forwarded to TuneRequest.input_path.
        **tuner_kwargs:
            Any scalar TuneRequest fields (n_trials, seed, eps_low, ...).
            Unknown keys raise TypeError immediately (before scheduling).
        """
        if self.is_running(dataset):
            logger.warning(
                "Tuning job for dataset %r is already in-flight — ignoring duplicate launch.",
                dataset,
            )
            return

        request = TuneRequest(input_path=Path(input_path), **tuner_kwargs)
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._run(dataset, request),
            name=f"tune:{dataset}",
        )
        self._running[dataset] = task

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _run(self, dataset: str, request: TuneRequest) -> None:
        """Background coroutine: offload run_tuning to the thread pool."""
        logger.info("Tuning started for dataset %r (n_trials=%d).", dataset, request.n_trials)
        loop = asyncio.get_running_loop()
        try:
            result: TuneResult = await loop.run_in_executor(None, run_tuning, request)
        except Exception as exc:
            logger.exception("Unexpected error while tuning dataset %r: %s", dataset, exc)
            return
        finally:
            self._running.pop(dataset, None)

        if result.status != "ok":
            logger.error(
                "Tuning failed for dataset %r: [%s] %s",
                dataset,
                result.error_code,
                result.error_message,
            )
            return

        try:
            params = _result_to_tuned_params(dataset, result)
            self._store.set(dataset, params)
        except Exception as exc:
            logger.exception("Failed to persist tuning result for dataset %r: %s", dataset, exc)
            return

        logger.info(
            "Tuning complete for dataset %r: score=%.4f, eps=%.4f, k=%d, topk=%d.",
            dataset,
            params.score,
            params.eps,
            params.k,
            params.topk,
        )
