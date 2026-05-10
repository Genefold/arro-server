"""
get_query_emb.py
Generates query embeddings from queries.json using nomic-embed-text-v1.5.

Outputs:
  data/nomic_embs/queries_emb_768.npy    → (n_entries, 768)  float32, L2-normalised
  data/nomic_embs/queries_emb_512.npy    → (n_entries, 512)  float32, L2-normalised
  data/nomic_embs/queries_emb_256.npy    → (n_entries, 256)  float32, L2-normalised
  data/nomic_embs/queries_index.json     → query_id → {row_index, expected_prompt_id,
                                                        query_text, query_type}
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT        = Path("/content/drive/MyDrive/prompt_kaban")
OUTPUT_DIR  = ROOT / "results"                   # same folder as corpus embeddings
QUERY_PATH  = ROOT / "queries.json"
OUT_INDEX   = OUTPUT_DIR / "queries_index.json"

MODEL_ID    = "nomic-ai/nomic-embed-text-v1.5"
QUERY_PREFIX = "search_query: "                  # required by Nomic for queries
BATCH_SIZE  = 512
ALL_DIMS    = [768, 512, 256]                    # Matryoshka slices


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading queries from {QUERY_PATH}...")
    with open(QUERY_PATH, "r") as f:
        query_corpus = json.load(f)

    n = len(query_corpus)
    print(f"  {n} queries loaded.\n")

    sentences     = []
    index_mapping = {}

    for idx, entry in enumerate(query_corpus):
        query_id    = entry["query_id"]
        query_text  = entry["query_text"]
        expected_id = entry["expected_prompt_id"]
        query_type  = entry.get("query_type", "unknown")   # ← preserved

        sentences.append(QUERY_PREFIX + query_text)

        index_mapping[query_id] = {
            "row_index":          idx,
            "expected_prompt_id": expected_id,
            "query_text":         query_text,
            "query_type":         query_type,   # ← now in the index
        }

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    # ── Encode ─────────────────────────────────────────────────────────────────
    print(f"Encoding {len(sentences)} queries (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )   # shape: (n, 768), float32

    # ── Save per dimension (Matryoshka truncation + L2 normalise) ─────────────
    print()
    for dim in ALL_DIMS:
        sliced = embeddings[:, :dim].copy()

        # Re-normalise after truncation so cosine similarity stays valid
        norms  = np.linalg.norm(sliced, axis=1, keepdims=True)
        normed = (sliced / norms).astype(np.float32)

        out_path = OUTPUT_DIR / f"queries_emb_{dim}.npy"
        np.save(out_path, normed)
        print(f"  ✓ queries_emb_{dim}.npy  shape={normed.shape}  dtype={normed.dtype}")

    # ── Save index ─────────────────────────────────────────────────────────────
    with open(OUT_INDEX, "w") as f:
        json.dump(index_mapping, f, indent=4)

    print(f"\n  ✓ queries_index.json  ({n} entries)")
    print(f"\n{'━'*60}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Load hint  : emb = np.load('queries_emb_768.npy').astype(np.float64)")
    print(f"               idx = json.load(open('queries_index.json'))")
    print(f"               row = idx['q_01']['row_index']   # → int")
    print(f"               expected = idx['q_01']['expected_prompt_id']  # → 'pk_...'")
    print(f"{'━'*60}")


if __name__ == "__main__":
    main()