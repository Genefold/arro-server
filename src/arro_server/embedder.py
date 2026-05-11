"""EmbedderService — NL query -> 768-d nomic embedding.

Wraps sentence-transformers (nomic-ai/nomic-embed-text-v1.5, SDPA backend)
and exposes a single .embed(text) method that returns a L2-normalised
768-dimensional float64 numpy array ready to pass straight to
PromptSearchEngine.search().

The singleton is loaded once at application startup via the FastAPI lifespan
and cached for the lifetime of the process.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

_DIM = 768
# nomic requires this trust_remote_code=True flag and the task prefix.
_TASK_PREFIX = "search_query: "


class EmbedderService:
    """Thin wrapper around SentenceTransformer for nomic-embed-text-v1.5.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to ``nomic-ai/nomic-embed-text-v1.5``.
        Override via ARRO_SERVER_EMBEDDER_MODEL env var.
    """

    _instance: "EmbedderService | None" = None

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for NL search. "
                "Install with: pip install 'arro-server[nlp]'"
            ) from exc

        log.info("Loading embedder model: %s", model_name)
        self._model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            backend="sdpa",  # scaled-dot-product attention — faster, same results
        )
        self.model_name = model_name
        self.dim = _DIM
        log.info("EmbedderService ready — model=%s dim=%d", model_name, _DIM)

    # ── public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single NL query string.

        Returns a L2-normalised 768-d float64 numpy array.
        The nomic task prefix (``search_query: ``) is prepended automatically.
        """
        prefixed = _TASK_PREFIX + text.strip()
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].astype(np.float64)
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of NL query strings (shape: [N, 768], float64, L2-normed)."""
        prefixed = [_TASK_PREFIX + t.strip() for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float64)
        return vecs

    # ── singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "EmbedderService":
        if cls._instance is None:
            from .settings import get_settings
            cls._instance = cls(get_settings().embedder_model)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton — used in tests."""
        cls._instance = None
