"""Persistence layer for per-dataset tuned ArrowSpace graph parameters.

Uses a single JSON file keyed by dataset name.  Thread-safe for a
single-process server; not safe for concurrent multi-process writers.

Graph parameter names match pyarrowspace.ArrowSpaceBuilder.with_lambda_graph(
    eps, k, topk, p, sigma
).
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class TunedParams:
    eps: float      # neighbourhood radius / graph connectivity threshold
    k: int          # lambda-graph neighbour count
    topk: int       # retrieval neighbour count at query time
    p: float        # Minkowski p-norm
    sigma: float    # RBF kernel bandwidth
    score: float    # tuning objective score (e.g. recall@k)
    tuned_at: str   # ISO 8601 UTC timestamp
    dataset: str    # dataset key

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        if self.topk > self.k:
            raise ValueError(f"topk ({self.topk}) must be <= k ({self.k})")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.p <= 0:
            raise ValueError(f"p must be > 0, got {self.p}")
        if self.sigma is None or self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be numeric, got {type(self.score)}")
        if not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        try:
            datetime.fromisoformat(self.tuned_at)
        except ValueError as exc:
            raise ValueError(
                f"tuned_at is not a valid ISO 8601 timestamp: {self.tuned_at!r}"
            ) from exc

    @staticmethod
    def now_utc() -> str:
        return datetime.now(tz=UTC).isoformat()

    @classmethod
    def from_dict(cls, data: dict) -> TunedParams:
        return cls(
            eps=float(data["eps"]),
            k=int(data["k"]),
            topk=int(data["topk"]),
            p=float(data["p"]),
            sigma=float(data["sigma"]),
            score=float(data["score"]),
            tuned_at=str(data["tuned_at"]),
            dataset=str(data["dataset"]),
        )

    def to_graph_params(self) -> dict:
        """Return a dict ready for pyarrowspace.ArrowSpaceBuilder.with_lambda_graph()."""
        return {"eps": self.eps, "k": self.k, "topk": self.topk,
                "p": self.p, "sigma": self.sigma}


class TuneStore:
    """Persist and retrieve TunedParams keyed by dataset name.

    The backing file is a JSON object ``{dataset_name: {…fields…}}``.
    Created on first write if absent. The parent directory must exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def get(self, dataset: str) -> TunedParams | None:
        """Return TunedParams for *dataset*, or None if not found."""
        raw = self._load().get(dataset)
        return TunedParams.from_dict(raw) if raw is not None else None

    def set(self, dataset: str, params: TunedParams) -> None:
        """Persist *params* for *dataset*, overwriting any prior entry."""
        if params.dataset != dataset:
            raise ValueError(
                f"params.dataset ({params.dataset!r}) does not match key ({dataset!r})"
            )
        with self._lock:
            store = self._load()
            store[dataset] = asdict(params)
            self._persist(store)

    def delete(self, dataset: str) -> bool:
        """Remove entry for *dataset*. Returns True if it existed."""
        with self._lock:
            store = self._load()
            existed = dataset in store
            if existed:
                del store[dataset]
                self._persist(store)
        return existed

    def all(self) -> dict[str, TunedParams]:
        """Return all stored params as {dataset: TunedParams}."""
        return {k: TunedParams.from_dict(v) for k, v in self._load().items()}

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"TuneStore backing file is corrupt ({self._path}): {exc}"
            ) from exc

    def _persist(self, store: dict) -> None:
        """Atomic write via tmp-file rename (POSIX-safe)."""
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
        tmp.replace(self._path)
