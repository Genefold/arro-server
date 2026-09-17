"""Pydantic request/response schemas for ArrowSpace endpoints.

Using explicit Pydantic models instead of dict[str, Any] for all POST
bodies ensures FastAPI validates inputs and returns 422 automatically for
missing or wrongly-typed fields — before the route body ever runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Keys that belong to the ArrowSpaceBuilder graph_params dict.
# If an incoming IndexBuildRequest body contains ONLY these keys (flat),
# we hoist the whole body into {"graph_params": <body>} automatically.
_GRAPH_PARAM_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})


class SearchEnergyRequest(BaseModel):
    """Body for POST /datasets/{id}/search/energy.

    Real arrowspace signature: search_energy(vec, gl, k)
    """

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchHybridRequest(BaseModel):
    """Body for POST /datasets/{id}/search/hybrid.

    Real arrowspace signature: search_hybrid(vec, gl, alpha)
    Note: 'tau' is NOT a parameter of the real search_hybrid implementation.
    """

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor (0=spectral, 1=linear).")


class SearchLinearRequest(BaseModel):
    """Body for POST /datasets/{id}/search/linear.

    Real arrowspace signature: search_linear_sorted(vec, gl, k)
    """

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchBatchRequest(BaseModel):
    """Body for POST /datasets/{id}/search/batch."""

    vectors: list[list[float]] = Field(..., description="Batch of query vectors.")
    tau: float = Field(1.0, description="Taumode tau parameter.")


class IndexBuildRequest(BaseModel):
    """Optional body for POST /datasets/{id}/index.

    Accepts two equivalent shapes:

    Structured (canonical)::

        {"graph_params": {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}}

    Flat (convenience — the whole body is treated as graph_params)::

        {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
    """

    graph_params: dict[str, Any] | None = Field(
        default=None,
        description="ArrowSpaceBuilder graph params. Omit to use server defaults.",
    )

    @model_validator(mode="before")
    @classmethod
    def _hoist_flat_graph_params(cls, values: Any) -> Any:
        """If the body is a flat dict of graph-param keys, wrap it."""
        if not isinstance(values, dict):
            return values
        # Already structured — has "graph_params" key or is empty
        if "graph_params" in values or not values:
            return values
        # All keys are known graph-param keys → flat payload
        if values.keys() <= _GRAPH_PARAM_KEYS:
            return {"graph_params": values}
        return values


class SearchModeRequest(BaseModel):
    """Body for POST /datasets/{id}/search (unified search with mode selector)."""

    vector: list[float] = Field(..., description="Query vector.")
    mode: Literal["taumode", "hybrid", "energy", "linear_sorted"] = Field(
        "taumode", description="Search mode: taumode | hybrid | energy | linear_sorted"
    )
    tau: float = Field(1.0, description="Tau param for taumode.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Blend for hybrid mode.")
    k: int = Field(10, ge=1, description="Top-k for energy and linear_sorted.")


# ---------------------------------------------------------------------------
# Upload schemas (#21)
# ---------------------------------------------------------------------------


class UploadInitRequest(BaseModel):
    """Request body for POST /api/upload/init.

    The client provides the intended dataset_id and the root label under
    which the dataset will live.  The server validates both and returns the
    absolute filesystem path where the client should write the Zarr array.

    Attributes:
        dataset_id: URL-safe dataset ID in the form ``<root>--<path>``.
                    E.g. ``"main--my_embeddings"``.
                    Must not contain path separators (``/``, ``\\``).
        root:       Root label that must exist in ARRO_SERVER_DATA_ROOTS.
                    E.g. ``"main"``.
    """

    dataset_id: str = Field(
        ...,
        description="URL-safe dataset ID, e.g. 'main--my_embeddings'.",
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_\-]+$",
    )
    root: str = Field(
        ...,
        description="Root label that must exist in ARRO_SERVER_DATA_ROOTS.",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-]+$",
    )


class UploadInitResponse(BaseModel):
    """Response body for POST /api/upload/init.

    Attributes:
        dataset_id:  The validated dataset ID, echoed back.
        upload_path: Absolute filesystem path where the client must write
                     the Zarr v3 array directory.  The path is guaranteed
                     to be inside a configured data root.
        root:        Root label confirmed by the server.
    """

    dataset_id: str
    upload_path: str
    root: str


class UploadCommitRequest(BaseModel):
    """Request body for POST /api/upload/commit.

    The client calls this after writing the Zarr array to upload_path.
    The server validates the path, opens the Zarr node, and inserts the
    dataset into the StorageRegistry cache via register_dataset().

    Attributes:
        dataset_id: The dataset ID returned by /upload/init.
        fs_path:    The upload_path returned by /upload/init.
                    The server re-validates this against resolved_roots
                    before any filesystem access (path-traversal guard).
    """

    dataset_id: str = Field(..., min_length=1, max_length=256)
    fs_path: str = Field(..., min_length=1, max_length=4096)


class UploadCommitResponse(BaseModel):
    """Response body for POST /api/upload/commit.

    Attributes:
        dataset_id:    The registered dataset ID.
        registered:    Always True on success.
        shape:         Shape of the registered array, e.g. [1000, 128].
        dtype:         Dtype string, e.g. "float32".
        chunks:        Chunk shape if available, else None.
        index_stale:   True if an ArrowSpace index already existed for this
                       dataset_id before the commit (i.e. the dataset was
                       overwritten).  The client should rebuild the index via
                       POST /datasets/{id}/index if True.
    """

    dataset_id: str
    registered: bool = True
    shape: list[int]
    dtype: str
    chunks: list[int] | None = None
    index_stale: bool = False


class VectorAppendRequest(BaseModel):
    """Request body for POST /api/datasets/{dataset_id}/vectors/append.

    Attributes:
        vectors: List of row-vectors, shape (M, D). M > 0. D must match
                 the target dataset's feature dimension.
        dtype:   Optional target dtype string (e.g. "float32"). If omitted
                 the array's current dtype is used. A mismatch raises 422.
    """

    vectors: list[list[float]]
    dtype: str | None = None


class VectorAppendResponse(BaseModel):
    """Response body for POST /api/datasets/{dataset_id}/vectors/append.

    Attributes:
        start_row:    Row index of the first appended vector (= old row count).
        appended:     Number of vectors written (= len(request.vectors)).
        new_shape:    Updated array shape [new_nrows, D].
        index_stale:  True if an ArrowSpace index exists for this dataset and
                      is now out of date (the append added rows not reflected
                      in the index). The client should call
                      POST /datasets/{id}/index to rebuild.
    """

    start_row: int
    appended: int
    new_shape: list[int]
    index_stale: bool = False


class RowUpdate(BaseModel):
    """A single row-overwrite instruction.

    Attributes:
        row_index: Zero-based row index to overwrite. Must be >= 0.
                   Out-of-bounds against the array's current shape is
                   validated by the backend (VectorShapeMismatch → 422).
        vector:    Replacement vector. Length must equal the dataset's
                   feature dimension D.
    """

    row_index: int = Field(..., ge=0)
    vector: list[float]


class VectorOverwriteRequest(BaseModel):
    """Request body for POST /api/datasets/{dataset_id}/vectors/overwrite.

    Attributes:
        updates: List of (row_index, vector) pairs. Must be non-empty.
                 Duplicate row_index values are permitted — the last entry
                 for a given index wins. No deduplication is performed.
        dtype:   Optional target dtype string (e.g. "float32"). If omitted,
                 defaults to "float64" at the route level. A same_kind cast
                 is attempted; incompatible dtype raises 422.
    """

    updates: list[RowUpdate] = Field(..., min_length=1)
    dtype: str | None = None


class VectorOverwriteResponse(BaseModel):
    """Response body for POST /api/datasets/{dataset_id}/vectors/overwrite.

    Attributes:
        overwritten:  Number of rows written (= len(request.updates)).
                      Shape of the array is unchanged.
        index_stale:  True if an ArrowSpace index exists for this dataset and
                      is now out of date (the overwritten vectors are no longer
                      reflected in the index). The client should call
                      POST /datasets/{id}/index to rebuild.
    """

    overwritten: int
    index_stale: bool = False


class VectorCountResponse(BaseModel):
    """Response body for GET /api/datasets/{dataset_id}/vectors/count.

    Reflects the StorageRegistry cache at the moment of the call.
    See staleness note in StorageRegistry.get_dataset().

    Attributes:
        dataset_id: Echo of the requested dataset ID.
        nrows:      Current row count (arr.shape[0]).
        ncols:      Feature dimension (arr.shape[1]).
    """

    dataset_id: str
    nrows: int
    ncols: int


# ---------------------------------------------------------------------------
# Tune schemas (Issue #66)
# ---------------------------------------------------------------------------


class TuneRequest(BaseModel):
    """Body for POST /datasets/{id}/tune.

    All ranges are inclusive [low, high].  Omitting a range lets the
    tuner use its own defaults (eps_low=0.5, eps_high=12.0, k_low=10,
    k_high=45).

    Attributes:
        dataset:  Dataset ID to tune against.
        eps_range: Optional [low, high] sweep for eps (graph connectivity
                    threshold).  Both values must be > 0 and low <= high.
        k_range:   Optional [low, high] sweep for k (lambda-graph neighbour
                    count).  Both values must be >= 1 and low <= high.
        n_trials:  Number of Optuna/hyperopt trials.  Clamped to [1, 500].
    """

    dataset: str = Field(
        ...,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_\-]+$",
        description="Dataset ID to tune against.",
    )
    eps_range: tuple[float, float] | None = Field(
        default=None,
        description="Inclusive [low, high] sweep for eps. Both values must be > 0 and low <= high.",
    )
    k_range: tuple[int, int] | None = Field(
        default=None,
        description="Inclusive [low, high] sweep for k. Both values must be >= 1 and low <= high.",
    )
    n_trials: int = Field(
        default=30,
        ge=1,
        le=500,
        description="Number of Optuna/hyperopt trials.",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> TuneRequest:
        if self.eps_range is not None:
            low, high = self.eps_range
            if low <= 0:
                raise ValueError(f"eps_range: low bound must be > 0, got {low}")
            if low > high:
                raise ValueError(f"eps_range: low ({low}) must be <= high ({high})")
        if self.k_range is not None:
            low, high = self.k_range
            if low < 1:
                raise ValueError(f"k_range: low bound must be >= 1, got {low}")
            if low > high:
                raise ValueError(f"k_range: low ({low}) must be <= high ({high})")
        return self


class TunedParamsSchema(BaseModel):
    """Serializable representation of a TunedParams dataclass.

    Mirrors arro_server.storage.tune_store.TunedParams (eps/k/topk/p/sigma
    lambda-graph parameters).  Validated on construction so the API never
    surfaces a corrupt store entry as a 200 response.

    Attributes:
        eps:      Neighbourhood radius / graph connectivity threshold.  Must be > 0.
        k:        Lambda-graph neighbour count.  Must be >= 1.
        topk:     Retrieval neighbour count at query time.  1 <= topk <= k.
        p:        Minkowski p-norm.  Must be > 0.
        sigma:    RBF kernel bandwidth, or None (allowed per tuner contract).
        score:    Tuning objective score (0.0 - 1.0).
        tuned_at: ISO 8601 UTC timestamp of when tuning completed.
        dataset:  Dataset this result belongs to.
    """

    eps: float = Field(..., gt=0.0)
    k: int = Field(..., ge=1)
    topk: int = Field(..., ge=1)
    p: float = Field(..., gt=0.0)
    sigma: float | None = Field(default=None, gt=0.0)
    score: float = Field(..., ge=0.0, le=1.0)
    tuned_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp, e.g. '2026-09-17T14:00:00+00:00'.",
    )
    dataset: str = Field(..., min_length=1)

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def _validate_cross_field(self) -> TunedParamsSchema:
        if self.topk > self.k:
            raise ValueError(f"topk ({self.topk}) must be <= k ({self.k})")
        try:
            datetime.fromisoformat(self.tuned_at)
        except ValueError as exc:
            raise ValueError(
                f"tuned_at is not a valid ISO 8601 timestamp: {self.tuned_at!r}"
            ) from exc
        return self


class TuneStatusResponse(BaseModel):
    """Response for GET /datasets/{id}/tune/status.

    Attributes:
        dataset: Dataset ID this status belongs to.
        status:  Current tuning state.
                  'not_started' — no tune run has been recorded.
                  'running'     — a tune job is currently in progress.
                  'done'        — tuning has completed; params is populated.
        params:  Tuned parameters.  Present only when status == 'done'.
    """

    dataset: str
    status: Literal["running", "done", "not_started"]
    params: TunedParamsSchema | None = None

    @model_validator(mode="after")
    def _params_present_iff_done(self) -> TuneStatusResponse:
        if self.status == "done" and self.params is None:
            raise ValueError("params must be set when status is 'done'")
        if self.status != "done" and self.params is not None:
            raise ValueError(f"params must be None when status is '{self.status}'")
        return self
