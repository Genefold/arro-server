"""Security helpers for arro-server API handlers.

Currently contains path-traversal protection for upload endpoints.
Add future auth / rate-limiting helpers here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def assert_path_within_roots(fs_path: Path, roots: dict[str, Path]) -> None:
    """Raise HTTP 400 if fs_path is not inside any configured data root.

    This is the primary path-traversal guard for upload endpoints.  A client
    that controls the ``fs_path`` field in UploadCommitRequest could otherwise
    pass ``../../etc/passwd`` or an absolute path outside the data roots.

    The check uses ``Path.is_relative_to()`` after resolving both paths to
    their canonical forms (``Path.resolve()``), eliminating symlink and
    ``..`` tricks.

    Args:
        fs_path: Candidate path supplied by the client (already resolved
                 by the caller via ``Path(raw).expanduser().resolve()``).
        roots:   Mapping of root label -> resolved root Path, as returned
                 by ``Settings.resolved_roots``.

    Raises:
        HTTPException(400): if fs_path is not inside any configured root,
            with a detail message that lists root labels (not paths) to
            avoid leaking server filesystem layout.

    Example::

        roots = {"main": Path("/data/zarr")}
        assert_path_within_roots(Path("/data/zarr/cube.zarr"), roots)  # OK
        assert_path_within_roots(Path("/etc/passwd"), roots)           # raises 400
        assert_path_within_roots(Path("/data/zarr/../../../etc"), roots)  # raises 400
    """
    resolved = fs_path.resolve()
    for root_path in roots.values():
        resolved_root = root_path.resolve()
        try:
            resolved.relative_to(resolved_root)
            return
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail=(
            f"fs_path is not inside any configured data root. "
            f"Configured roots: {list(roots.keys())}. "
            f"Ensure the path was obtained from POST /upload/init."
        ),
    )


def validate_zarr_summary(dataset_id: str, shape: tuple[int, ...], dtype: str) -> None:
    """Raise HTTP 422 if the Zarr summary indicates a corrupt or empty array.

    Called after ``ZarrFilesystemBackend.summarize()`` to catch the case where
    the client called ``/upload/commit`` before the filesystem had fully flushed
    the Zarr write (NFS buffering, incomplete upload).

    Args:
        dataset_id: Dataset ID, used in error messages only.
        shape:      Shape tuple from the DatasetSummary.
        dtype:      Dtype string from the DatasetSummary.

    Raises:
        HTTPException(422): if shape is empty (scalar), all-zero, or dtype
            is an empty string — all indicators of an incomplete Zarr write.
    """
    if not shape:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{dataset_id}' has scalar shape () — "
                "the Zarr array may be incomplete. "
                "Ensure the write is fully flushed before calling /upload/commit."
            ),
        )
    if any(dim == 0 for dim in shape):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{dataset_id}' has zero-size shape {list(shape)} — "
                "the Zarr array appears empty. "
                "Ensure the array contains data before calling /upload/commit."
            ),
        )
    if not dtype:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset '{dataset_id}' has empty dtype — "
                "the Zarr metadata may be incomplete. "
                "Ensure zarr.json is fully written before calling /upload/commit."
            ),
        )
