"""Pydantic request/response schemas for ArrowSpace endpoints.

Using explicit Pydantic models instead of dict[str, Any] for all POST
bodies ensures FastAPI validates inputs and returns 422 automatically for
missing or wrongly-typed fields — before the route body ever runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

# Keys that belong to the ArrowSpaceBuilder graph_params dict.
# If an incoming IndexBuildRequest body contains ONLY these keys (flat),
# we hoist the whole body into {"graph_params": <body>} automatically.
_GRAPH_PARAM_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})


class SearchRequest(BaseModel):
    """Body for POST /datasets/{id}/search (spectral taumode search)."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    tau: float = Field(1.0, description="Taumode tau parameter.")


class SearchEnergyRequest(BaseModel):
    """Body for POST /datasets/{id}/search/energy."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchHybridRequest(BaseModel):
    """Body for POST /datasets/{id}/search/hybrid."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor (0=spectral, 1=linear).")


class SearchLinearRequest(BaseModel):
    """Body for POST /datasets/{id}/search/linear."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchBatchRequest(BaseModel):
    """Body for POST /datasets/{id}/search/batch."""

    vectors: list[list[float]] = Field(..., description="Batch of query vectors.")
    tau: float = Field(1.0, description="Taumode tau parameter.")


class PromptSearchRequest(BaseModel):
    """Body for POST /api/prompts/search.

    The caller supplies a pre-computed 768-d nomic embedding.
    Use POST /api/prompts/nl_search to let the server embed the query.
    """

    vector: list[float] = Field(..., description="768-dim nomic-embed-text-v1.5 query vector.")
    k: int              = Field(10, ge=1, le=100, description="Number of results to return.")
    tau: float          = Field(0.75, ge=0.0, le=5.0, description="Spectral sharpness (0=broad, 5=sharp). Default 0.75.")
    alpha: float        = Field(0.6, ge=0.0, le=1.0, description="Cosine vs spectral blend.")
    lam: float          = Field(0.7, ge=0.0, le=1.0, description="MMR diversity weight (1.0=pure relevance, 0.0=max diversity).")

    @model_validator(mode="after")
    def _check_vector_dim(self) -> "PromptSearchRequest":
        if len(self.vector) != 768:
            raise ValueError(f"vector must have exactly 768 dimensions, got {len(self.vector)}")
        return self


class NLSearchRequest(BaseModel):
    """Body for POST /api/prompts/nl_search.

    The server embeds `query` using EmbedderService and runs the search.
    This is the primary endpoint for frontend consumers.
    """

    query: str   = Field(..., min_length=1, max_length=2048, description="Natural language search query.")
    k: int       = Field(10, ge=1, le=100, description="Number of results to return.")
    tau: float   = Field(0.75, ge=0.0, le=5.0, description="Spectral sharpness. Default 0.75.")
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Cosine vs spectral blend.")
    lam: float   = Field(0.7, ge=0.0, le=1.0, description="MMR diversity weight.")


class PromptSearchResult(BaseModel):
    """A single result returned by /prompts/search or /prompts/nl_search.

    The dataset stores the prompt text in the ``content`` field.
    ``body`` is kept as a read-only alias so older clients keep working.

    Scoring fields use plain names (score, salience, tau) so Pydantic v2
    serialises them correctly.  AliasChoices also accepts the legacy
    underscore-prefixed variants (_score, _salience, _tau) for any callers
    that still produce those keys.
    """

    id: str
    title: str | None = None
    # Primary field — matches the 'content' key in dataset.json
    content: str | None = None
    # Alias for backward compat — returns the same value as content
    body: str | None = None
    tags: list[str] = Field(default_factory=list)
    upvotes: int | None = None
    views: int | None = None
    author_reputation: float | None = None
    version: int | None = None
    fork_count: int | None = None
    likes: int | None = None
    downvotes: int | None = None
    uses: int | None = None
    created_at: str | None = None
    category: str | None = None
    subcategory: str | None = None
    has_placeholders: bool | None = None
    placeholders: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str | None = None
    target_model: str | None = None

    # Scoring fields — plain names so Pydantic v2 includes them in output.
    # AliasChoices accepts both the canonical name and the legacy _-prefixed
    # variant so nothing breaks if old code still produces "_score" keys.
    score: float = Field(
        default=0.0,
        validation_alias=AliasChoices("score", "_score"),
        description="MMR-reranked relevance score.",
    )
    salience: float = Field(
        default=0.0,
        validation_alias=AliasChoices("salience", "_salience"),
        description="Salience score (upvotes/likes/reputation/views blend).",
    )
    tau: float = Field(
        default=0.0,
        validation_alias=AliasChoices("tau", "_tau"),
        description="Spectral tau used for this result.",
    )

    @model_validator(mode="after")
    def _sync_body_content(self) -> "PromptSearchResult":
        """Keep body and content in sync — whichever is set populates the other."""
        if self.content is not None and self.body is None:
            self.body = self.content
        elif self.body is not None and self.content is None:
            self.content = self.body
        return self

    model_config = {"extra": "allow", "populate_by_name": True}


class PromptSearchResponse(BaseModel):
    """Response envelope for /prompts/search and /prompts/nl_search."""

    query: str | None = Field(None, description="Original NL query (nl_search only).")
    k: int
    tau: float
    lam: float
    results: list[PromptSearchResult]
    result_count: int


class IndexBuildRequest(BaseModel):
    """Optional body for POST /datasets/{id}/index.

    Accepts two equivalent shapes:

    Structured (canonical)::

        {"graph_params": {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}}

    Flat (convenience -- the whole body is treated as graph_params)::

        {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
    """

    graph_params: dict[str, Any] = Field(
        default_factory=dict,
        description="ArrowSpace build parameters (eps, k, topk, p, sigma).",
    )

    @model_validator(mode="before")
    @classmethod
    def _hoist_flat_params(cls, data: Any) -> Any:
        """If the body contains only graph-param keys (flat form), wrap them."""
        if isinstance(data, dict):
            if "graph_params" not in data and data.keys() <= _GRAPH_PARAM_KEYS:
                return {"graph_params": data}
        return data
