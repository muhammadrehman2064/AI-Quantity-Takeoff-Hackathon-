# AI Architectural Quantity Takeoff — Hackathon MVP

Architectural PDF → AI drawing reading → visible dimensions → preliminary quantities → simple BOQ → review.

## Pipeline (4 small AI calls)

| # | Agent | Input | Output (compact JSON) | max_tokens (qwen) |
|---|-------|-------|-----------------------|-------------------|
| 1 | Drawing (vision) | page images + PDF text hints | summary, per-page type/level/scale, visible dimensions, elements | 800 |
| 2 | Measurement (text) | Agent 1 dimensions only (no images) | `measurements[]` + `unverified_items[]` | 700 |
| 3 | QS | measurements + 3 short methodology extracts | `items[]`, assumptions, excluded_items | 850 |
| 4 | Review | BOQ + measurements + automatic findings | only the problem items + warnings | 500 |

Free Groq limits `qwen/qwen3.8-27b` to **1000 output tokens/minute**, so every request stays under 850
and the code paces output tokens per model. Agents 2–4 default to `openai/gpt-oss-120b`, which has its own
separate limit.

Review also runs free, deterministic checks in Python (missing formula/page, formula arithmetic,
duplicates, low confidence). If the AI review call fails, the BOQ is still produced with these checks.

## Error handling

* JSON replies are parsed tolerantly (code fences, `<think>`, extra text, trailing commas); truncated
  JSON is repaired back to the last complete item and flagged with a warning.
* HTTP 429: oversized/daily-quota errors fail immediately with a clear message; temporary per-minute
  limits are retried at most twice. The SDK's own hidden retries are disabled.
* Errors appear as messages in the UI, never tracebacks. If a late stage fails, earlier results stay visible.

## Files

- `app.py` — Streamlit UI and workflow
- `ai_agents.py` — Groq calls, token pacing, JSON parsing, the four agents
- `drawing_processor.py` — PDF → compact JPEGs (+ dimension text hints from the PDF text layer)
- `quantity_engine.py` — automatic checks, review merge, BOQ assembly
- `rag.py` — methodology retrieval from the prebuilt `rag_index/` (no indexing or embedding at runtime)
- `build_index.py`, `make_query_vector.py` — optional offline tools

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a Groq API key in the sidebar, or set `GROQ_API_KEY` as an environment variable / Streamlit secret.

## RAG

`rag_index/` holds a prebuilt FAISS index (313 chunks) of measurement-rule text. The app uses a fixed
query, so no embedding model is needed in production:

* With `rag_index/query_vector.npy` present → FAISS search.
* Without it → keyword ranking over the stored chunk text (current default).

To enable the FAISS path, run once on a local machine (not on Streamlit Cloud) and commit the result:

```bash
pip install sentence-transformers
python make_query_vector.py
```

(`build_index.py` also writes the vector when you rebuild the index from a `knowledge_base/` folder.)

## Limitations

Preliminary takeoff assistant only. Dimensions are never invented; unreadable values are listed as
unverified. Scanned or low-resolution drawings give weaker results; check quantities against the originals.
