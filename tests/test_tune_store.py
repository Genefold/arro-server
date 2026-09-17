"""Tests for TuneStore and TunedParams — ArrowSpace graph parameters."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from arro_server.storage.tune_store import TunedParams, TuneStore


@pytest.fixture
def store(tmp_path: Path) -> TuneStore:
    return TuneStore(tmp_path / "tune_params.json")


def make_params(dataset: str = "mnist", **overrides) -> TunedParams:
    defaults = dict(
        eps=0.5, k=10, topk=3, p=2.0, sigma=1.0,
        score=0.92,
        tuned_at="2026-09-17T14:00:00+00:00",
        dataset=dataset,
    )
    return TunedParams(**(defaults | overrides))


# --- Validation ---

class TestTunedParamsValidation:
    def test_valid(self):
        p = make_params()
        assert p.eps == 0.5 and p.k == 10

    def test_k_zero_invalid(self):
        with pytest.raises(ValueError, match="k must be"):
            make_params(k=0)

    def test_topk_zero_invalid(self):
        with pytest.raises(ValueError, match="topk must be"):
            make_params(topk=0)

    def test_topk_greater_than_k_invalid(self):
        with pytest.raises(ValueError, match=r"topk.*<=.*k"):
            make_params(k=5, topk=10)

    def test_eps_zero_invalid(self):
        with pytest.raises(ValueError, match="eps"):
            make_params(eps=0.0)

    def test_eps_negative_invalid(self):
        with pytest.raises(ValueError, match="eps"):
            make_params(eps=-0.1)

    def test_sigma_zero_invalid(self):
        with pytest.raises(ValueError, match="sigma"):
            make_params(sigma=0.0)

    def test_sigma_none_invalid(self):
        with pytest.raises(ValueError, match="sigma"):
            make_params(sigma=None)  # type: ignore

    def test_p_zero_invalid(self):
        with pytest.raises(ValueError, match="p must be"):
            make_params(p=0.0)

    def test_empty_dataset_invalid(self):
        with pytest.raises(ValueError, match="dataset"):
            make_params(dataset="")

    def test_invalid_tuned_at(self):
        with pytest.raises(ValueError, match="ISO 8601"):
            make_params(tuned_at="yesterday")

    def test_score_as_int_accepted(self):
        assert make_params(score=1).score == 1.0

    def test_score_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            make_params(score="high")  # type: ignore

    def test_from_dict_round_trip(self):
        p = make_params()
        assert TunedParams.from_dict(p.__dict__) == p  # type: ignore

    def test_to_graph_params_keys(self):
        gp = make_params().to_graph_params()
        assert set(gp) == {"eps", "k", "topk", "p", "sigma"}

    def test_now_utc_is_valid_iso(self):
        from datetime import datetime
        ts = TunedParams.now_utc()
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


# --- CRUD ---

class TestTuneStoreCRUD:
    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_set_and_get(self, store):
        p = make_params("cifar")
        store.set("cifar", p)
        assert store.get("cifar") == p

    def test_set_overwrites(self, store):
        store.set("cifar", make_params("cifar", score=0.80))
        store.set("cifar", make_params("cifar", score=0.95))
        assert store.get("cifar").score == 0.95

    def test_mismatched_key_raises(self, store):
        with pytest.raises(ValueError, match="does not match key"):
            store.set("cifar", make_params("mnist"))

    def test_multiple_datasets_independent(self, store):
        store.set("mnist", make_params("mnist", score=0.90))
        store.set("cifar", make_params("cifar", score=0.80))
        assert store.get("mnist").score == 0.90
        assert store.get("cifar").score == 0.80

    def test_all_returns_all(self, store):
        store.set("mnist", make_params("mnist"))
        store.set("cifar", make_params("cifar"))
        assert set(store.all().keys()) == {"mnist", "cifar"}

    def test_all_empty(self, store):
        assert store.all() == {}

    def test_delete_existing_returns_true(self, store):
        store.set("mnist", make_params("mnist"))
        assert store.delete("mnist") is True
        assert store.get("mnist") is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False

    def test_delete_does_not_affect_other_keys(self, store):
        store.set("mnist", make_params("mnist"))
        store.set("cifar", make_params("cifar"))
        store.delete("mnist")
        assert store.get("cifar") is not None


# --- Persistence ---

class TestTuneStorePersistence:
    def test_file_not_created_on_read(self, tmp_path):
        path = tmp_path / "tune.json"
        TuneStore(path).get("any")
        assert not path.exists()

    def test_file_created_on_write(self, tmp_path):
        path = tmp_path / "tune.json"
        TuneStore(path).set("mnist", make_params("mnist"))
        assert path.exists()

    def test_survives_reinstantiation(self, tmp_path):
        path = tmp_path / "tune.json"
        TuneStore(path).set("mnist", make_params("mnist", eps=0.7))
        assert TuneStore(path).get("mnist").eps == 0.7

    def test_json_is_human_readable(self, tmp_path):
        path = tmp_path / "tune.json"
        TuneStore(path).set("mnist", make_params("mnist"))
        data = json.loads(path.read_text())
        assert "mnist" in data
        assert "eps" in data["mnist"]

    def test_corrupt_file_raises_runtime_error(self, tmp_path):
        path = tmp_path / "tune.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(RuntimeError, match="corrupt"):
            TuneStore(path).get("any")

    def test_empty_file_treated_as_empty(self, tmp_path):
        path = tmp_path / "tune.json"
        path.write_text("", encoding="utf-8")
        assert TuneStore(path).get("any") is None


# --- Concurrency ---

class TestTuneStoreConcurrency:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        path = tmp_path / "tune.json"
        store = TuneStore(path)
        errors: list[Exception] = []

        def write(ds: str) -> None:
            try:
                for _ in range(20):
                    store.set(ds, make_params(ds))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(f"ds{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.all()) == 5
