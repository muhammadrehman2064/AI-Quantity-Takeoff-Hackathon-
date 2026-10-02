# AI Architectural Quantity Takeoff — MVP

A 2-day hackathon prototype that uses a multi-agent workflow to analyse architectural
PDF drawings and produce a preliminary quantity takeoff / BOQ.

## Architecture

PDF
→ Drawing Agent
→ Measurement Agent
→ FAISS/RAG methodology
→ QS Agent
→ Independent Review Agent
→ Preliminary BOQ

## Main files

- `app.py` — Streamlit UI and workflow
- `drawing_processor.py` — PDF rendering
- `agents.py` — Groq vision/text agents
- `quantity_engine.py` — BOQ assembly
- `rag.py` — FAISS retrieval
- `build_index.py` — one-time local index builder
- `knowledge_base/` — trusted QS reference documents
- `rag_index/` — generated FAISS index

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a Groq API key in the sidebar.

## Build the RAG index

Place PDF/TXT references inside `knowledge_base/`, then:

```bash
python build_index.py
```

Commit `rag_index/index.faiss` and `rag_index/metadata.json` if their size is
acceptable for your GitHub/Streamlit deployment.

## Important MVP limitation

This is a preliminary takeoff assistant, not a replacement for a QS or engineer.
The system is intentionally instructed not to invent dimensions. Values that cannot
be supported by drawing evidence should be flagged for review.

## Deployment

Push the repository to GitHub and deploy the repository on Streamlit Community Cloud.
Set `GROQ_API_KEY` as a Streamlit secret, or enter it in the sidebar.

## Hackathon demo

Recommended demo sequence:

1. Upload a real architectural PDF.
2. Show the drawing analysis.
3. Show extracted dimensions with page references/confidence.
4. Show the QS BOQ with formulas.
5. Show the independent review tab.
6. Explain the agentic pipeline and RAG methodology layer.
