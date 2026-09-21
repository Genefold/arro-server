"""Route handlers for launching and polling tuner jobs (issue #67).

Endpoints
---------
POST /api/datasets/{id}/tune   -> 202 {"status": "started" | "running", "dataset": ...}
                                  404 if the dataset does not exist.
GET  /api/datasets/{id}/tune   -> 200 TuneStatusResponse
                                  ("running" | "done" | "not_started").

Design notes:
- POST is serialized per dataset_id by an asyncio.Lock (issue #80): the
  is_running() guard, the .npy export and launch() run inside the lock, so
  two near-simultaneous POSTs cannot both pass the guard and export
  concurrently to the same .tmp.npy path.
- The tuner (arrowspace_tuner) only accepts .npy/.npz files, while datasets
  live in Zarr, so POST exports the full array to a stable .npy path once per
  launch before handing it to TunerAdapter.launch().  The file is regenerated
  on every launch, never read back by the API.
- Job status is derived: running if TunerAdapter.is_running(), else "done"
  when persisted params exist, else "not_started".  A failed job persists
  nothing, so it reports "not_started" (the adapter contract has no
  "failed" state).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_tuner_adapter
from ..settings import Settings, get_settings
from ..slicing import parse_slice
from ..storage import StorageRegistry, get_registry
from ..tuner_adapter import TunerAdapter
from .schemas import (
    TunedParamsSchema,
    TuneRequest,
    TuneStartResponse,
    TuneStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tuning"])

# Per-dataset serialization for POST /tune (issue #80): the guard→export→launch
# sequence must be atomic, else two concurrent POSTs both pass is_running()
# and export to the same .tmp.npy path.
# TODO(cleanup): _TUNE_LOCKS grows unboundedly with unique dataset IDs, like
# tune_inputs/<id>.npy; same follow-up issue tracks both.
_TUNE_LOCKS: dict[str, asyncio.Lock] = {}


def _get_tune_lock(dataset_id: str) -> asyncio.Lock:
    """Return the per-dataset lock, creating it on first use.

    Sync function with no await points: the check-and-insert is atomic under
    the single event loop, no meta-lock needed.
    """
    lock = _TUNE_LOCKS.get(dataset_id)
    if lock is None:
        lock = _TUNE_LOCKS[dataset_id] = asyncio.Lock()
    return lock


def _registry() -> StorageRegistry:
    return get_registry()


def _tune_input_path(settings: Settings, dataset_id: str) -> Path:
    """Return the stable .npy export path for *dataset_id* (safe filename)."""
    base = Path(settings.tune_params_path).expanduser().resolve().parent / "tune_inputs"
    return base / f"{dataset_id.replace('/', '--')}.npy"


def _export_dataset_to_npy(dataset_id: str, reg: StorageRegistry, settings: Settings) -> Path:
    """Export the full Zarr array to a .npy file for the tuner.

    Blocking I/O + O(N*D) read: must run via asyncio.to_thread.
    Raises DatasetNotFound when the dataset does not exist.
    """
    h = reg.open(dataset_id)
    if len(h.summary.shape) != 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{dataset_id}' is not 2-D (shape={h.summary.shape}); "
                "tuning requires a 2-D array."
            ),
        )
    rs = parse_slice(None, h.summary.shape, offset=0, limit=h.summary.shape[0])
    arr = h.read_window(rs)
    out = _tune_input_path(settings, dataset_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (tmp + replace, same pattern as TuneStore._persist):
    # defence-in-depth only — POST /tune is serialized per dataset by
    # _TUNE_LOCKS (issue #80), so concurrent exports no longer happen here.
    # TODO(cleanup): tune_inputs/<id>.npy is overwritten per launch but never
    # garbage-collected; delete stale entries in lifespan shutdown or track
    # in a follow-up issue.
    np.save(out.with_suffix(".tmp.npy"), np.asarray(arr, dtype=np.float64))
    out.with_suffix(".tmp.npy").replace(out)
    return out


@router.post(
    "/datasets/{dataset_id:path}/tune",
    status_code=202,
    response_model=TuneStartResponse,
    summary="Launch a background tuning job for a dataset",
)
async def start_tune(
    dataset_id: str,
    body: TuneRequest | None = None,
    reg: StorageRegistry = Depends(_registry),
    adapter: TunerAdapter = Depends(get_tuner_adapter),
    settings: Settings = Depends(get_settings),
) -> TuneStartResponse:
    """Launch a background tuning job for *dataset_id*.

    Idempotent: if a job is already in progress the call is a no-op and
    returns ``{"status": "running"}`` with the same 202 status code.
    Returns 404 if the dataset does not exist.
    """
    if body is not None and body.dataset != dataset_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body.dataset ('{body.dataset}') does not match the URL "
                f"dataset_id ('{dataset_id}')."
            ),
        )

    # Serialize guard → export → launch per dataset (issue #80).  asyncio.Lock
    # is awaitable: the second concurrent request suspends here instead of
    # racing into the export.  When it resumes, is_running() is True and it
    # returns "running" without a second launch.
    lock = _get_tune_lock(dataset_id)
    async with lock:
        if adapter.is_running(dataset_id):
            logger.info("Tune job for '%s' already in progress — returning running", dataset_id)
            return TuneStartResponse(status="running", dataset=dataset_id)

        # DatasetNotFound is itself an HTTPException(404) — let it propagate.
        npy_path = await asyncio.to_thread(_export_dataset_to_npy, dataset_id, reg, settings)
        tuner_kwargs: dict[str, object] = {"n_trials": body.n_trials if body else 30}
        if body is not None:
            if body.eps_range is not None:
                tuner_kwargs["eps_low"], tuner_kwargs["eps_high"] = body.eps_range
            if body.k_range is not None:
                tuner_kwargs["k_low"], tuner_kwargs["k_high"] = body.k_range

        adapter.launch(dataset_id, npy_path, **tuner_kwargs)
        logger.info("Tune job for '%s' launched (input=%s)", dataset_id, npy_path)
        return TuneStartResponse(status="started", dataset=dataset_id)


@router.get(
    "/datasets/{dataset_id:path}/tune",
    response_model=TuneStatusResponse,
    summary="Poll the status of a tuning job",
)
async def get_tune_status(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: TunerAdapter = Depends(get_tuner_adapter),
) -> TuneStatusResponse:
    """Return the current tuning job status for *dataset_id*.

    - "not_started" — no job has been submitted (or none succeeded).
    - "running"     — job is in progress; params is null.
    - "done"        — a previous job completed; params is populated.
    """
    if reg.get_dataset(dataset_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    if adapter.is_running(dataset_id):
        return TuneStatusResponse(dataset=dataset_id, status="running")

    persisted = adapter.get_params(dataset_id)
    if persisted is None:
        return TuneStatusResponse(dataset=dataset_id, status="not_started")

    return TuneStatusResponse(
        dataset=dataset_id,
        status="done",
        params=TunedParamsSchema.model_validate(persisted),
    )
