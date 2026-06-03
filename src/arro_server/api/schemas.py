"""Pydantic request/response schemas for ArrowSpace endpoints.

Using explicit Pydantic models instead of dict[str, Any] for all POST
bodies ensures FastAPI validates inputs and returns 422 automatically for
missing or wrongly-typed fields — before the route body ever runs.
"""

from __future__ import annotations

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
