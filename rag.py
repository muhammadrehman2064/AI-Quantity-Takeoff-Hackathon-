"""Methodology retrieval for the QS agent.

Uses the PREBUILT FAISS index in rag_index/. No documents are indexed and no embeddings are
generated while the app runs:

* If rag_index/query_vector.npy exists (made offline by make_query_vector.py) the fixed
  methodology query is searched in the FAISS index.
* Otherwise a tiny keyword ranking over the stored chunk text is used (pure Python).

If nothing is available, an explicit "no context" note is returned and the app still works.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

INDEX_DIR = Path(__file__).resolve().parent / "rag_index"

# One fixed query: the QS agent always needs the same kind of measurement guidance.
METHODOLOGY_QUERY = (
    "architectural quantity takeoff measurement rules BOQ floor area wall length "
    "door window count room dimensions"
)
_KEYWORDS = (
    "measure", "measured", "area", "floor", "wall", "door", "window", "net", "gross",
    "length", "number", "m2", "quantity", "dimension", "opening", "deduct",
)

_NO_RAG = [
    {
        "text": "No methodology index available. Use only supplied measurements; do not assume rules.",
        "source": "RAG unavailable",
    }
]


def load_rag() -> Optional[Dict[str, Any]]:
    """Load chunk metadata (+ FAISS index and query vector when available). Returns None if absent."""
    metadata_file = INDEX_DIR / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        chunks = json.loads(metadata_file.read_text(encoding="utf-8")).get("chunks", [])
    except Exception:
        return None
    if not chunks:
        return None

    rag: Dict[str, Any] = {"chunks": chunks, "index": None, "query_vector": None}
    index_file = INDEX_DIR / "index.faiss"
    vector_file = INDEX_DIR / "query_vector.npy"
    if index_file.exists() and vector_file.exists():
        try:
            import faiss
            import numpy as np

            index = faiss.read_index(str(index_file))
            vector = np.asarray(np.load(str(vector_file)), dtype="float32").reshape(1, -1)
            if index.ntotal == len(chunks) and vector.shape[1] == index.d:
                rag["index"], rag["query_vector"] = index, vector
        except Exception:
            pass  # fall back to keyword ranking
    return rag


def _keyword_rank(chunks: List[dict], top_k: int) -> List[int]:
    scored = []
    for i, chunk in enumerate(chunks):
        text = (chunk.get("text") or "").lower()
        if len(text) < 200:
            continue
        hits = sum(text.count(k) for k in _KEYWORDS)
        scored.append((hits / (len(text) ** 0.5), i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:top_k]]


def retrieve_methodology(rag: Optional[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Return up to top_k methodology chunks for the fixed QS query."""
    if not rag:
        return list(_NO_RAG)

    chunks = rag["chunks"]
    ids: List[int] = []
    if rag.get("index") is not None:
        try:
            _, found = rag["index"].search(rag["query_vector"], top_k)
            ids = [int(i) for i in found[0] if 0 <= int(i) < len(chunks)]
        except Exception:
            ids = []
    if not ids:
        ids = _keyword_rank(chunks, top_k)

    results = [
        {
            "text": chunks[i].get("text", ""),
            "source": chunks[i].get("source", ""),
            "page": chunks[i].get("page_number"),
        }
        for i in ids
    ]
    return results or list(_NO_RAG)
