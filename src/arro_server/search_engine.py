# src/arro_server/search_engine.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from arrowspace import ArrowSpaceBuilder

_ROOT     = Path(__file__).parents[2]
_EMBS_DIR = _ROOT / "data" / "nomic_embs"
_DATASET  = _ROOT / "data" / "dataset.json"
_DIM      = 768
W_UP, W_LK, W_REP, W_VIEW = 0.35, 0.35, 0.20, 0.10
SAL_WEIGHT = 0.30
LAM        = 1.0
DEFAULT_TAU = 0.75


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def _load_best_params() -> dict:
    """Load eps and k from the latest tuner run. tau is intentionally excluded
    — it is a runtime parameter controlled by the caller, not a corpus constant."""
    tuner_dir = _ROOT / "notebooks" / "results" / "arrowspace_tuner"
    latest    = sorted(tuner_dir.iterdir())[-1] / "best_params.json"
    raw       = json.loads(latest.read_text())
    params    = raw.get("params", raw)
    # strip tau so ArrowSpaceBuilder only receives graph-topology params
    return {k: v for k, v in params.items() if k != "tau"}


class PromptSearchEngine:
    _instance: "PromptSearchEngine | None" = None

    def __init__(self) -> None:
        # 1. embeddings + ids
        embs_path = _EMBS_DIR / f"embeddings_nomic_structured_{_DIM}d_raw.npy"
        ids_path  = _EMBS_DIR / f"embeddings_nomic_structured_{_DIM}d_ids.npy"
        self.embs: np.ndarray = np.load(embs_path).astype(np.float64)
        self.ids:  list[str]  = list(np.load(ids_path))
        assert self.embs.shape[0] == len(self.ids)

        # 2. dataset map: pk_NNNNN → full JSON record
        dataset: list[dict] = json.loads(_DATASET.read_text())
        self.dataset_map: dict[str, dict] = {r["id"]: r for r in dataset}

        # 3. salience (parallel to embs rows)
        records    = [self.dataset_map[pk] for pk in self.ids]
        upvotes    = _norm(np.array([r.get("upvotes", 0)           for r in records], dtype=float))
        likes      = _norm(np.array([r.get("likes", 0)             for r in records], dtype=float))
        reputation = _norm(np.array([r.get("author_reputation", 0) for r in records], dtype=float))
        views      = _norm(np.log1p(np.array([r.get("views", 0)    for r in records], dtype=float)))
        sal_arr    = _norm(W_UP * upvotes + W_LK * likes + W_REP * reputation + W_VIEW * views)
        self.salience: dict[str, float] = {
            self.ids[i]: float(sal_arr[i]) for i in range(len(self.ids))
        }

        # 4. build aspace + gl with tuner-optimised eps & k (no tau)
        self.aspace, self.gl = ArrowSpaceBuilder().build(_load_best_params(), self.embs)

    @classmethod
    def get(cls) -> "PromptSearchEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── search ────────────────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        tau: float = DEFAULT_TAU,
        alpha: float = 0.6,
    ) -> list[dict]:
        q = np.asarray(query_vec, dtype=np.float64).ravel()
        candidates: list[tuple[int, float]] = self.aspace.search(q, k=k * 3, tau=tau, alpha=alpha)
        reranked = self._mmr(candidates, k)
        out = []
        for row_idx, score in reranked:
            pk     = self.ids[row_idx]
            record = dict(self.dataset_map[pk])
            record["_score"]    = round(score, 6)
            record["_salience"] = round(self.salience.get(pk, 0.0), 6)
            record["_tau"]      = tau
            out.append(record)
        return out

    def _mmr(self, candidates: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        def rel(i: int) -> float:
            ri, cos = candidates[i]
            return (1 - SAL_WEIGHT) * cos + SAL_WEIGHT * self.salience.get(self.ids[ri], 0.0)

        selected, remaining = [], list(range(len(candidates)))
        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=rel)
            else:
                sel_embs = np.array([self.embs[candidates[i][0]] for i in selected])
                def mmr(i: int) -> float:
                    e   = self.embs[candidates[i][0]]
                    sim = np.max(sel_embs @ e / (np.linalg.norm(sel_embs, axis=1) * np.linalg.norm(e) + 1e-9))
                    return LAM * rel(i) - (1 - LAM) * sim
                best = max(remaining, key=mmr)
            selected.append(best)
            remaining.remove(best)
        return [candidates[i] for i in selected]
