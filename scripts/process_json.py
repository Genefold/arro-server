import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "corpus_embs" / "dataset.json"




# ── helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Remove {{placeholder}} tokens, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"\{\{[^}]+\}\}", "", text)).strip()


def is_weak_title(el: dict) -> bool:
    """True when title has ≤4 words or shares no token with its own tags/category."""
    t = el["title"].lower()
    signals = el.get("tags", []) + [el.get("category", ""), el.get("subcategory", "")]
    has_signal = any(s.lower().replace("-", " ") in t for s in signals if s)
    return len(el["title"].split()) <= 4 or not has_signal
