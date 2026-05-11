# src/arro_server/search_engine.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder

log = logging.getLogger(__name__)

_DIM = 768

W_UP, W_LK, W_REP, W_VIEW = 0.35, 0.35, 0.20, 0.10
SAL_WEIGHT  = 0.30
LAM         = 0.7   # MMR diversity weight: 1.0=pure relevance, 0.0=max diversity
DEFAULT_TAU = 0.75

# Keys that ArrowSpaceBuilder.build() accepts at graph-construction time.
# tau is a *query-time* parameter and must never be passed to build().
_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})

# Hardcoded fallback params used when tuner results are absent (fresh clone / container).
# Values come from the tuner run documented in the notebook (search_engine_demo.ipynb).
_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0}


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def _load_best_params(data_dir: Path) -> dict:
    """Load graph build params from the latest tuner run.

    Falls back to _DEFAULT_BUILD_PARAMS when the tuner output directory is
    missing (fresh clone, container, CI).  tau is stripped because it is a
    query-time parameter and must not be passed to ArrowSpaceBuilder.build().
    """
    tuner_dir = data_dir.parent / "notebooks" / "results" / "arrowspace_tuner"
    try:
        candidates = sorted(tuner_dir.iterdir())
        if not candidates:
            raise FileNotFoundError("no tuner run directories found")
        latest = candidates[-1] / "best_params.json"
        raw    = json.loads(latest.read_text())
        params = raw.get("params", raw)
        filtered = {k: v for k, v in params.items() if k in _BUILD_KEYS}
        if not filtered:
            raise ValueError("best_params.json contained no recognised build keys")
        log.info("Loaded ArrowSpace build params from tuner: %s", filtered)
        return filtered
    except (FileNotFoundError, IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "Tuner results unavailable (%s) — using hardcoded defaults: %s",
            exc,
            _DEFAULT_BUILD_PARAMS,
        )
        return dict(_DEFAULT_BUILD_PARAMS)


class PromptSearchEngine:
    """Singleton that holds the ArrowSpace index and exposes semantic search.

    The data directory is resolved from Settings.prompt_data_dir so the
    engine works correctly both in development layout and inside containers.
    """

    _instance: "PromptSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        embs_path    = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_raw.npy"
        ids_path     = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_ids.npy"
        dataset_path = data_dir / "dataset.json"

        for p in (embs_path, ids_path, dataset_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"PromptSearchEngine: required file missing: {p}\n"
                    f"Set ARRO_SERVER_PROMPT_DATA_DIR to the directory containing "
                    f"dataset.json and nomic_embs/."
                )

        # 1. embeddings + ids
        self.embs: np.ndarray = np.load(embs_path).astype(np.float64)
        self.ids:  list[str]  = list(np.load(ids_path, allow_pickle=True))
        if self.embs.shape[0] != len(self.ids):
            raise AssertionError(
                f"embeddings rows ({self.embs.shape[0]}) != id count ({len(self.ids)})"
            )

        # 2. dataset map  pk_NNNNN -> full JSON record
        dataset: list[dict] = json.loads(dataset_path.read_text())
        self.dataset_map: dict[str, dict] = {r["id"]: r for r in dataset}

        # 3. salience vector (parallel to embs rows)
        records    = [self.dataset_map[pk] for pk in self.ids]
        upvotes    = _norm(np.array([r.get("upvotes", 0)           for r in records], dtype=float))
        likes      = _norm(np.array([r.get("likes", 0)             for r in records], dtype=float))
        reputation = _norm(np.array([r.get("author_reputation", 0) for r in records], dtype=float))
        views      = _norm(np.log1p(np.array([r.get("views", 0)    for r in records], dtype=float)))
        sal_arr    = _norm(W_UP * upvotes + W_LK * likes + W_REP * reputation + W_VIEW * views)
        self.salience: dict[str, float] = {
            self.ids[i]: float(sal_arr[i]) for i in range(len(self.ids))
        }

        # 4. build graph index — eps & k only, tau excluded
        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)
        log.info(
            "PromptSearchEngine ready — nitems=%d nfeatures=%d nclusters=%d",
            self.aspace.nitems,
            self.aspace.nfeatures,
            self.aspace.nclusters,
        )

    @classmethod
    def get(cls) -> "PromptSearchEngine":
        if cls._instance is None:
            from .settings import get_settings
            data_dir = Path(get_settings().prompt_data_dir).resolve()
            cls._instance = cls(data_dir)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton — used in tests."""
        cls._instance = None

    # ── public search ─────────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        k: int       = 10,
        tau: float   = DEFAULT_TAU,
        alpha: float = 0.6,
        lam: float   = LAM,
    ) -> list[dict]:
        """Run spectral search and MMR rerank.

        Parameters
        ----------
        query_vec : 768-d float64 nomic embedding (already L2-normalised by caller)
        k         : number of final results after MMR rerank
        tau       : spectral sharpness passed to aspace.search(vec, gl, tau)
        alpha     : cosine-vs-salience blend (kept for _mmr); not an aspace param
        lam       : MMR diversity weight (1.0=pure relevance, 0.0=max diversity)
        """
        q = np.asarray(query_vec, dtype=np.float64).ravel()
        if q.shape[0] != _DIM:
            raise ValueError(f"query_vec must be {_DIM}-dimensional, got {q.shape[0]}")

        # aspace.search signature: (vec, gl, tau) — three positional args, no kwargs
        raw_candidates: list[tuple[int, float]] = self.aspace.search(q, self.gl, tau)
        candidates = raw_candidates[: k * 3]
        reranked   = self._mmr(candidates, k, lam)

        out = []
        for row_idx, score in reranked:
            pk     = self.ids[row_idx]
            record = dict(self.dataset_map[pk])
            record["_score"]    = round(score, 6)
            record["_salience"] = round(self.salience.get(pk, 0.0), 6)
            record["_tau"]      = tau
            out.append(record)
        return out

    # ── MMR reranker ──────────────────────────────────────────────────────────
    def _mmr(
        self,
        candidates: list[tuple[int, float]],
        k: int,
        lam: float,
    ) -> list[tuple[int, float]]:
        """Maximal Marginal Relevance with salience boosting.

        rel(i)   = (1 - SAL_WEIGHT) * cosine_score + SAL_WEIGHT * salience
        mmr(i)   = lam * rel(i) - (1 - lam) * max_sim_to_selected

        lam=1.0  -> pure relevance (no diversity penalty)
        lam=0.0  -> pure diversity (ignore relevance)
        Default  -> 0.7 (balances relevance and diversity)
        """
        def rel(i: int) -> float:
            row_idx, cos_score = candidates[i]
            sal = self.salience.get(self.ids[row_idx], 0.0)
            return (1 - SAL_WEIGHT) * cos_score + SAL_WEIGHT * sal

        selected:  list[int] = []
        remaining: list[int] = list(range(len(candidates)))

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=rel)
            else:
                sel_embs = np.array([self.embs[candidates[i][0]] for i in selected])

                def mmr_score(i: int, _sel: np.ndarray = sel_embs) -> float:
                    e   = self.embs[candidates[i][0]]
                    nrm = np.linalg.norm(_sel, axis=1) * np.linalg.norm(e) + 1e-9
                    sim = float(np.max(_sel @ e / nrm))
                    return lam * rel(i) - (1 - lam) * sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]
