#!/usr/bin/env python3
"""
nomic_corpus_embs.py

Reads pre-processed doc_strings from TinyDB, generates Matryoshka embeddings
using nomic-ai/nomic-embed-text-v1.5, and persists both .npy arrays and an
id-to-row index in a dedicated TinyDB registry for bullet-proof ID matching.

Outputs per requested dim (under OUTPUT_DIR):
    embeddings_nomic_structured_{dim}d_raw.npy   float32  (N, dim)
    embeddings_nomic_structured_{dim}d_ids.npy   unicode  (N,)   ← parallel ID array

TinyDB registry (OUTPUT_DIR / embeddings_registry.json):
    one record per (id, dim) → {id, dim, row_idx, npy_path}

Usage:
    python nomic_corpus_embs.py                 # all dims: 768, 512, 256
    python nomic_corpus_embs.py --dims 768
    python nomic_corpus_embs.py --dims 512
    python nomic_corpus_embs.py --dims 256
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tinydb import TinyDB, Query

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/content/drive/MyDrive/prompt_kaban")
SOURCE_DB   = ROOT / "db.json"
OUTPUT_DIR  = ROOT / "results" 
REGISTRY_DB = OUTPUT_DIR / "embeddings_registry.json"

MODEL_ID   = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 256
ALL_DIMS   = [768, 512, 256]


# ── load records ───────────────────────────────────────────────────────────────
def load_records(db_path: Path) -> tuple[list[str], list[str]]:
    """
    Opens the source TinyDB, validates every record, deduplicates,
    and returns (ids, texts) sorted by id for fully deterministic row order.
    Raises loudly on any integrity issue so bad data can never silently
    corrupt the embedding array.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Source DB not found: {db_path}")

    db  = TinyDB(db_path)
    raw = db.all()
    db.close()

    if not raw:
        raise ValueError(f"Source DB is empty: {db_path}")

    # field validation
    for i, rec in enumerate(raw):
        if "id" not in rec:
            raise KeyError(f"Record at DB index {i} is missing 'id': {rec}")
        if "doc_string" not in rec:
            raise KeyError(f"Record id={rec['id']!r} is missing 'doc_string'")
        if not isinstance(rec["doc_string"], str) or not rec["doc_string"].strip():
            raise ValueError(f"Record id={rec['id']!r} has empty doc_string")

    # sort by id → same row order every run, regardless of insertion order
    raw.sort(key=lambda r: r["id"])

    ids   = [r["id"]         for r in raw]
    texts = [r["doc_string"] for r in raw]

    # duplicate ID guard
    if len(set(ids)) != len(ids):
        from collections import Counter
        dupes = [i for i, n in Counter(ids).items() if n > 1]
        raise ValueError(f"Duplicate IDs detected in source DB: {dupes}")

    return ids, texts


# ── encoding ───────────────────────────────────────────────────────────────────
def encode_corpus(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encodes all texts in batches → unnormalised float32 (N, 768).
    Full 768d is produced once; each Matryoshka dim is sliced from it.
    """
    t0 = time.perf_counter()
    print(f"  Encoding {len(texts):,} documents...", flush=True)
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s   shape={vecs.shape}", flush=True)
    return vecs.astype(np.float32)


def matryoshka_slice(vecs: np.ndarray, d: int) -> np.ndarray:
    return vecs[:, :d].astype(np.float32, copy=True)


# ── save + register ────────────────────────────────────────────────────────────
def save_and_register(
    ids: list[str],
    raw_vecs: np.ndarray,
    d: int,
    output_dir: Path,
    registry: TinyDB,
) -> None:
    """
    1. Slices the (N, 768) array to (N, d).
    2. Saves the embedding .npy AND a parallel IDs .npy in the same row order.
    3. Round-trip verifies both files before touching the registry.
    4. Bulk-replaces the registry records for this dim (remove + insert_multiple)
       so the registry always reflects the current files exactly.
    """
    n = len(ids)
    sliced  = matryoshka_slice(raw_vecs, d)          # (N, d)
    ids_arr = np.array(ids, dtype="U64")             # unicode, same row order

    emb_path = output_dir / f"embeddings_nomic_structured_{d}d_raw.npy"
    ids_path = output_dir / f"embeddings_nomic_structured_{d}d_ids.npy"

    np.save(emb_path, sliced)
    np.save(ids_path, ids_arr)

    # ── round-trip integrity check ──────────────────────────────────────────
    loaded_emb = np.load(emb_path)
    loaded_ids = np.load(ids_path)

    assert loaded_emb.shape == (n, d), (
        f"Embedding shape mismatch after save: expected ({n}, {d}), "
        f"got {loaded_emb.shape}"
    )
    assert len(loaded_ids) == n, (
        f"ID array length mismatch after save: expected {n}, got {len(loaded_ids)}"
    )
    assert list(loaded_ids) == ids, (
        "ID round-trip FAILED — saved IDs do not match source IDs. "
        "Do NOT use these files."
    )
    assert np.allclose(loaded_emb, sliced, atol=0), (
        "Embedding round-trip FAILED — values changed after save."
    )

    print(f"  ✓  {emb_path.name}   shape={sliced.shape}   dtype={sliced.dtype}")
    print(f"  ✓  {ids_path.name}   entries={len(ids_arr)}   verified ✓")

    # ── bulk registry update ────────────────────────────────────────────────
    # Remove all existing records for this dim, then re-insert in one call.
    # This is O(1) TinyDB writes instead of O(N), safe for 20k+ records.
    Rec = Query()
    registry.remove(Rec.dim == d)
    registry.insert_multiple([
        {
            "id":       pid,
            "dim":      d,
            "row_idx":  idx,
            "npy_path": str(emb_path),
            "ids_path": str(ids_path),
        }
        for idx, pid in enumerate(ids)
    ])
    print(f"  ✓  Registry updated: {n:,} records for dim={d}")


# ── CLI + main ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Matryoshka embeddings from TinyDB doc_strings."
    )
    p.add_argument(
        "--dims",
        default="all",
        help="Matryoshka dims to produce: 768 | 512 | 256 | all (default: all)",
    )
    p.add_argument(
        "--source-db",
        default=str(SOURCE_DB),
        help=f"Path to source TinyDB (default: {SOURCE_DB})",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for .npy files and registry (default: {OUTPUT_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    dims_list = ALL_DIMS if args.dims == "all" else [int(args.dims)]
    src_db    = Path(args.source_db)
    out_dir   = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. load ───────────────────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  SOURCE DB : {src_db}")
    print(f"  OUTPUT DIR: {out_dir}")
    print(f"  DIMS      : {dims_list}")
    print(f"{'━'*60}\n")

    print(f"Reading source DB...", flush=True)
    ids, texts = load_records(src_db)
    print(f"  {len(ids):,} records loaded (sorted by id)\n")

    print(f"  Sample record:")
    print(f"    id   = {ids[0]!r}")
    print(f"    text = {texts[0][:140]!r}\n")

    # ── 2. model ──────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} ...", flush=True)
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    print("  Model ready.\n")

    # ── 3. encode once at full 768d ───────────────────────────────────────────
    print("Encoding corpus at full 768d...", flush=True)
    raw_vecs = encode_corpus(model, texts)   # (N, 768)
    print()

    # ── 4. slice + save + register per dim ────────────────────────────────────
    print(f"Saving embeddings & updating registry...\n", flush=True)
    registry = TinyDB(out_dir / "embeddings_registry.json")

    for d in dims_list:
        print(f"  dim = {d}d", flush=True)
        save_and_register(ids, raw_vecs, d, out_dir, registry)
        print()

    registry.close()

    # ── 5. final summary ──────────────────────────────────────────────────────
    print(f"{'━'*60}")
    print(f"  All done.")
    print(f"  Registry : {out_dir / 'embeddings_registry.json'}")
    print(f"{'━'*60}\n")
    print("IMPORTANT — when loading into ArrowSpaceBuilder always cast to float64:")
    for d in dims_list:
        print(f"  embs = np.load('embeddings_nomic_structured_{d}d_raw.npy').astype(np.float64)")
        print(f"  ids  = np.load('embeddings_nomic_structured_{d}d_ids.npy')")
        print()
    print("Registry lookup example:")
    print("  from tinydb import TinyDB, Query")
    print("  reg = TinyDB('embeddings_registry.json')")
    print("  R   = Query()")
    print("  row = reg.search((R.id == 'pk_03138') & (R.dim == 768))[0]")
    print("  emb = np.load(row['npy_path'])[row['row_idx']]")


if __name__ == "__main__":
    main()