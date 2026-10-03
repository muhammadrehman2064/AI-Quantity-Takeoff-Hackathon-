"""One-time, offline helper: embed the fixed methodology query so the deployed app can search
the prebuilt FAISS index without loading an embedding model.

Usage (local machine, not Streamlit Cloud):
    pip install sentence-transformers numpy
    python make_query_vector.py
Then commit rag_index/query_vector.npy.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import INDEX_DIR, METHODOLOGY_QUERY

meta = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
model = SentenceTransformer(meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
vector = model.encode([METHODOLOGY_QUERY], normalize_embeddings=True)
np.save(INDEX_DIR / "query_vector.npy", np.asarray(vector, dtype="float32"))
print("Saved", INDEX_DIR / "query_vector.npy")
