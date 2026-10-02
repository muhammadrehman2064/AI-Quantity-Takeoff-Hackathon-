import json
from pathlib import Path
from typing import Any, Dict, List

INDEX_DIR = Path("rag_index")


def load_rag():
    """Load prebuilt FAISS/SentenceTransformer index if present.

    The app remains usable without an index; it simply returns no RAG context.
    """
    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "metadata.json"

    if not index_file.exists() or not metadata_file.exists():
        return None

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        model_name = metadata.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        model = SentenceTransformer(model_name)

        return {"index": index, "metadata": metadata, "model": model}
    except Exception:
        return None


def retrieve_methodology(rag, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not rag:
        return [
            {
                "text": "No prebuilt RAG index is installed. Do not use unsupported assumptions.",
                "source": "RAG index unavailable",
            }
        ]

    import numpy as np

    vector = rag["model"].encode([query], normalize_embeddings=True)
    distances, indices = rag["index"].search(
        np.asarray(vector, dtype="float32"), top_k
    )

    chunks = rag["metadata"].get("chunks", [])
    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "score": float(distance),
            }
        )

    return results
