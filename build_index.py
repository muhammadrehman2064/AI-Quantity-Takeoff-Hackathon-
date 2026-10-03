"""
Build a local FAISS index from text/PDF knowledge documents.

Usage:
    1. Put QS reference PDFs/TXT files into knowledge_base/
    2. pip install -r requirements.txt
    3. python build_index.py

The resulting rag_index/ folder can be committed to GitHub if it is small enough.
For large indexes, use Git LFS or another storage mechanism.
"""

import json
from pathlib import Path

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB = Path("knowledge_base")
OUT = Path("rag_index")
OUT.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
OVERLAP = 150


def extract_text(path: Path):
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    return ""


def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(len(words), start + CHUNK_SIZE)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - OVERLAP)

    return chunks


records = []

if not KB.is_dir():
    raise SystemExit("knowledge_base/ folder not found. Create it and add PDF/TXT documents first.")

for path in sorted(KB.iterdir()):
    if path.suffix.lower() not in {".pdf", ".txt"}:
        continue

    text = extract_text(path)
    for chunk_no, chunk in enumerate(chunk_text(text)):
        if chunk.strip():
            records.append(
                {
                    "text": chunk,
                    "source": path.name,
                    "chunk_no": chunk_no,
                }
            )

if not records:
    raise SystemExit(
        "No PDF/TXT knowledge documents found in knowledge_base/. Add documents first."
    )

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    [r["text"] for r in records],
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings, dtype="float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(OUT / "index.faiss"))
(OUT / "metadata.json").write_text(
    json.dumps(
        {
            "embedding_model": MODEL_NAME,
            "chunks": records,
            "total_vectors": int(index.ntotal),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Pre-embed the fixed methodology query so the deployed app never needs the embedding model.
from rag import METHODOLOGY_QUERY  # noqa: E402

query_vector = model.encode([METHODOLOGY_QUERY], normalize_embeddings=True)
np.save(OUT / "query_vector.npy", np.asarray(query_vector, dtype="float32"))

print(f"Created {index.ntotal} vectors in {OUT}/")
