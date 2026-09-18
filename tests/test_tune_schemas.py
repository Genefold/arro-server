"""Tests for TuneRequest, TunedParamsSchema, TuneStatusResponse."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arro_server.api.schemas import (
    TunedParamsSchema,
    TuneRequest,
    TuneStatusResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PARAMS = dict(
    eps=0.5,
    k=10,
    topk=3,
    p=2.0,
    sigma=1.0,
    score=0.97,
    tuned_at="2026-09-17T14:00:00+00:00",
    dataset="mnist",
)


def valid_request(**overrides) -> dict:
    base = dict(dataset="mnist", n_trials=30)
    return base | overrides


# ---------------------------------------------------------------------------
# TuneRequest
# ---------------------------------------------------------------------------


class TestTuneRequest:
    def test_minimal_valid(self):
        r = TuneRequest(**valid_request())
        assert r.dataset == "mnist"
        assert r.n_trials == 30
        assert r.eps_range is None
        assert r.k_range is None

    def test_with_all_ranges(self):
        r = TuneRequest(
            dataset="mnist",
            eps_range=(0.5, 12.0),
            k_range=(4, 64),
            n_trials=50,
        )
        assert r.eps_range == (0.5, 12.0)
        assert r.k_range == (4, 64)

    # --- dataset validation ---

    def test_empty_dataset_raises(self):
        with pytest.raises(ValidationError, match="dataset"):
            TuneRequest(dataset="", n_trials=30)

    def test_dataset_with_slash_raises(self):
        with pytest.raises(ValidationError):
            TuneRequest(dataset="bad/name")

    def test_dataset_too_long_raises(self):
        with pytest.raises(ValidationError):
            TuneRequest(dataset="a" * 257)

    # --- n_trials bounds ---

    def test_n_trials_zero_raises(self):
        with pytest.raises(ValidationError, match="n_trials"):
            TuneRequest(dataset="mnist", n_trials=0)

    def test_n_trials_above_500_raises(self):
        with pytest.raises(ValidationError, match="n_trials"):
            TuneRequest(dataset="mnist", n_trials=501)

    def test_n_trials_boundary_1(self):
        r = TuneRequest(dataset="mnist", n_trials=1)
        assert r.n_trials == 1

    def test_n_trials_boundary_500(self):
        r = TuneRequest(dataset="mnist", n_trials=500)
        assert r.n_trials == 500

    # --- eps_range validation ---

    def test_eps_range_low_zero_raises(self):
        with pytest.raises(ValidationError, match="eps_range"):
            TuneRequest(dataset="mnist", eps_range=(0.0, 12.0))

    def test_eps_range_negative_raises(self):
        with pytest.raises(ValidationError, match="eps_range"):
            TuneRequest(dataset="mnist", eps_range=(-1.0, 12.0))

    def test_eps_range_inverted_raises(self):
        with pytest.raises(ValidationError, match="eps_range"):
            TuneRequest(dataset="mnist", eps_range=(12.0, 0.5))

    def test_eps_range_equal_bounds_ok(self):
        r = TuneRequest(dataset="mnist", eps_range=(2.5, 2.5))
        assert r.eps_range == (2.5, 2.5)

    # --- k_range validation ---

    def test_k_range_low_zero_raises(self):
        with pytest.raises(ValidationError, match="k_range"):
            TuneRequest(dataset="mnist", k_range=(0, 64))

    def test_k_range_inverted_raises(self):
        with pytest.raises(ValidationError, match="k_range"):
            TuneRequest(dataset="mnist", k_range=(64, 4))

    def test_k_range_none_is_default(self):
        r = TuneRequest(dataset="mnist")
        assert r.k_range is None


# ---------------------------------------------------------------------------
# TunedParamsSchema
# ---------------------------------------------------------------------------


class TestTunedParamsSchema:
    def test_valid_construction(self):
        p = TunedParamsSchema(**VALID_PARAMS)
        assert p.score == 0.97

    def test_eps_zero_raises(self):
        with pytest.raises(ValidationError, match="eps"):
            TunedParamsSchema(**{**VALID_PARAMS, "eps": 0.0})

    def test_k_zero_raises(self):
        with pytest.raises(ValidationError, match="k"):
            TunedParamsSchema(**{**VALID_PARAMS, "k": 0})

    def test_topk_zero_raises(self):
        with pytest.raises(ValidationError, match="topk"):
            TunedParamsSchema(**{**VALID_PARAMS, "topk": 0})

    def test_topk_above_k_raises(self):
        with pytest.raises(ValidationError, match="topk"):
            TunedParamsSchema(**{**VALID_PARAMS, "topk": 11})

    def test_topk_equal_k_ok(self):
        p = TunedParamsSchema(**{**VALID_PARAMS, "topk": 10})
        assert p.topk == 10

    def test_p_zero_raises(self):
        with pytest.raises(ValidationError, match="p"):
            TunedParamsSchema(**{**VALID_PARAMS, "p": 0.0})

    def test_sigma_zero_raises(self):
        with pytest.raises(ValidationError, match="sigma"):
            TunedParamsSchema(**{**VALID_PARAMS, "sigma": 0.0})

    def test_sigma_none_ok(self):
        p = TunedParamsSchema(**{**VALID_PARAMS, "sigma": None})
        assert p.sigma is None

    def test_score_below_zero_raises(self):
        with pytest.raises(ValidationError, match="score"):
            TunedParamsSchema(**{**VALID_PARAMS, "score": -0.01})

    def test_score_above_one_allowed(self):
        # The tuner objective is 0.7*mrr + 0.2*log1p(fiedler) + 0.1*log1p(var)
        # — unbounded above 1.0. 2.137 is a real tuner output (#F1, issue #67
        # follow-up); the schema must not reject it.
        p = TunedParamsSchema(**{**VALID_PARAMS, "score": 2.137})
        assert p.score == 2.137

    def test_score_boundary_zero(self):
        p = TunedParamsSchema(**{**VALID_PARAMS, "score": 0.0})
        assert p.score == 0.0

    def test_score_boundary_one(self):
        p = TunedParamsSchema(**{**VALID_PARAMS, "score": 1.0})
        assert p.score == 1.0

    def test_empty_dataset_raises(self):
        with pytest.raises(ValidationError, match="dataset"):
            TunedParamsSchema(**{**VALID_PARAMS, "dataset": ""})

    def test_invalid_tuned_at_raises(self):
        with pytest.raises(ValidationError, match="ISO 8601"):
            TunedParamsSchema(**{**VALID_PARAMS, "tuned_at": "not-a-date"})

    def test_valid_tuned_at_no_tz(self):
        # Naive timestamps are valid ISO 8601 — accept them
        p = TunedParamsSchema(**{**VALID_PARAMS, "tuned_at": "2026-09-17T14:00:00"})
        assert "14:00:00" in p.tuned_at

    def test_from_attributes_orm_mode(self):
        """Verify from_attributes=True so TunedParams dataclass can be fed directly."""
        from arro_server.storage.tune_store import TunedParams

        tp = TunedParams(**VALID_PARAMS)
        schema = TunedParamsSchema.model_validate(tp, from_attributes=True)
        assert schema.eps == tp.eps
        assert schema.k == tp.k

    def test_json_serialization(self):
        p = TunedParamsSchema(**VALID_PARAMS)
        j = p.model_dump()
        assert j["score"] == 0.97
        assert j["dataset"] == "mnist"


# ---------------------------------------------------------------------------
# TuneStatusResponse
# ---------------------------------------------------------------------------


class TestTuneStatusResponse:
    def test_not_started_no_params(self):
        r = TuneStatusResponse(dataset="mnist", status="not_started")
        assert r.params is None

    def test_running_no_params(self):
        r = TuneStatusResponse(dataset="mnist", status="running")
        assert r.params is None

    def test_done_with_params(self):
        r = TuneStatusResponse(
            dataset="mnist",
            status="done",
            params=TunedParamsSchema(**VALID_PARAMS),
        )
        assert r.params.score == 0.97

    def test_done_without_params_raises(self):
        with pytest.raises(ValidationError, match="params must be set"):
            TuneStatusResponse(dataset="mnist", status="done", params=None)

    def test_not_started_with_params_raises(self):
        with pytest.raises(ValidationError, match="params must be None"):
            TuneStatusResponse(
                dataset="mnist",
                status="not_started",
                params=TunedParamsSchema(**VALID_PARAMS),
            )

    def test_running_with_params_raises(self):
        with pytest.raises(ValidationError, match="params must be None"):
            TuneStatusResponse(
                dataset="mnist",
                status="running",
                params=TunedParamsSchema(**VALID_PARAMS),
            )

    def test_invalid_status_literal_raises(self):
        with pytest.raises(ValidationError):
            TuneStatusResponse(dataset="mnist", status="pending")  # type: ignore[arg-type]

    def test_json_round_trip(self):
        r = TuneStatusResponse(
            dataset="mnist",
            status="done",
            params=TunedParamsSchema(**VALID_PARAMS),
        )
        j = r.model_dump()
        assert j["status"] == "done"
        assert j["params"]["eps"] == 0.5
