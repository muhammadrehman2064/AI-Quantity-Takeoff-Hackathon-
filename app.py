import json
import os

import streamlit as st

from ai_agents import (
    AgentError,
    run_drawing_agent,
    run_measurement_agent,
    run_qs_agent,
    run_review_agent,
)
from drawing_processor import process_drawing
from quantity_engine import build_boq, merge_reviews, precheck_boq
from rag import load_rag, retrieve_methodology

st.set_page_config(page_title="AI Quantity Takeoff", page_icon="📐", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag():
    """Prebuilt index only: nothing is indexed or embedded at runtime."""
    return load_rag()


def default_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = str(st.secrets.get("GROQ_API_KEY", ""))
        except Exception:
            key = ""
    return key


def show_error(exc: Exception) -> None:
    st.error(str(exc))
    detail = getattr(exc, "detail", "")
    if detail:
        with st.expander("Technical detail"):
            st.code(detail)


st.title("📐 AI Architectural Quantity Takeoff")
st.caption("PDF drawing → drawing analysis → visible dimensions → preliminary quantities / BOQ → review")

with st.sidebar:
    st.header("Project Settings")
    api_key = st.text_input("Groq API Key", type="password", value=default_api_key())
    vision_choice = st.selectbox("Vision model (Agent 1)", ["qwen/qwen3.8-27b", "Custom..."], index=0)
    vision_model = st.text_input("Custom vision model ID", "") if vision_choice == "Custom..." else vision_choice
    text_choice = st.selectbox(
        "Text model (Agents 2-4)",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "Custom..."],
        index=0,
    )
    text_model = st.text_input("Custom text model ID", "") if text_choice == "Custom..." else text_choice
    project_name = st.text_input("Project Name", "Sample Residential Project")
    unit_system = st.selectbox("Unit System", ["Metric", "Imperial"], index=0)
    run_pages = st.slider("Maximum PDF pages", 1, 3, 2)

uploaded = st.file_uploader("Upload architectural drawing PDF", type=["pdf"])

if "result" not in st.session_state:
    st.session_state.result = None

if uploaded:
    st.info(
        "Extracted measurements are preliminary. Check final quantities against the original drawings. "
        "Groq's free tier limits output tokens, so a run can take a few minutes."
    )

    if st.button("🚀 Run AI Quantity Takeoff", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
            st.stop()
        if not vision_model.strip() or not text_model.strip():
            st.error("Enter a model ID in the sidebar.")
            st.stop()

        result = {
            "project": project_name,
            "drawing_analysis": None,
            "measurements": None,
            "qs_result": None,
            "review": None,
            "boq": None,
            "warnings": [],
        }
        st.session_state.result = None

        try:
            with st.spinner("Rendering PDF pages..."):
                pages = process_drawing(uploaded.getvalue(), max_pages=run_pages)
            st.write(f"Pages prepared for analysis: **{len(pages)}**")

            with st.spinner("Agent 1/4 — analysing drawing and reading dimensions..."):
                result["drawing_analysis"] = run_drawing_agent(
                    api_key, vision_model.strip(), pages, project_name, unit_system
                )

            with st.spinner("Agent 2/4 — structuring measurements..."):
                result["measurements"] = run_measurement_agent(
                    api_key, text_model.strip(), result["drawing_analysis"], project_name, unit_system
                )

            methodology = retrieve_methodology(get_rag(), top_k=3)

            with st.spinner("Agent 3/4 — preparing QS quantities..."):
                result["qs_result"] = run_qs_agent(
                    api_key,
                    text_model.strip(),
                    result["drawing_analysis"],
                    result["measurements"],
                    methodology,
                    project_name,
                    unit_system,
                )

            auto = precheck_boq(result["measurements"], result["qs_result"])
            ai_review = None
            with st.spinner("Agent 4/4 — reviewing quantities..."):
                try:
                    ai_review = run_review_agent(
                        api_key, text_model.strip(), result["measurements"], result["qs_result"], auto
                    )
                except AgentError as exc:
                    result["warnings"].append(f"AI review skipped ({exc}). Automatic checks only.")

            result["review"] = merge_reviews(auto, ai_review)
            result["boq"] = build_boq(result["qs_result"], result["review"])
            st.success("Takeoff completed.")

        except (AgentError, ValueError) as exc:
            show_error(exc)
        except Exception as exc:  # never show a raw traceback to the user
            st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            # Keep whatever finished so a late failure does not lose earlier results.
            if result["drawing_analysis"] is not None:
                st.session_state.result = result

result = st.session_state.result

if result:
    for stage in ("drawing_analysis", "measurements", "qs_result"):
        data = result.get(stage) or {}
        if data.get("_warning"):
            st.warning(f"{stage.replace('_', ' ').title()}: {data['_warning']}")
    for w in result.get("warnings", []):
        st.warning(w)
    if result.get("boq") is None:
        st.warning("The run did not finish. Partial results are shown below; run again to complete.")

    tabs = st.tabs(["📋 Drawing", "📐 Measurements", "🧮 BOQ", "🔎 Review"])

    with tabs[0]:
        drawing = result["drawing_analysis"]
        st.subheader("Drawing Analysis")
        st.write(drawing.get("summary") or "No summary returned.")
        for page in drawing.get("pages", []):
            with st.expander(
                f"Page {page['page']} — {page['type'] or 'type unknown'} "
                f"(level: {page['level'] or '?'}, scale: {page['scale'] or '?'})",
                expanded=True,
            ):
                st.markdown("**Dimensions:** " + ("; ".join(page["dimensions"]) or "none read"))
                st.markdown("**Elements:** " + ("; ".join(page["elements"]) or "none"))
                if page["uncertainties"]:
                    st.markdown("**Uncertain:** " + "; ".join(page["uncertainties"]))

    with tabs[1]:
        st.subheader("Extracted Measurements")
        measurements = result.get("measurements")
        if measurements is None:
            st.info("Not available.")
        else:
            rows = measurements.get("measurements", [])
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No reliable numeric measurements were extracted.")
            if measurements.get("unverified_items"):
                st.markdown("**Unverified (not used as numbers):** " + "; ".join(measurements["unverified_items"]))

    with tabs[2]:
        st.subheader("Preliminary BOQ")
        boq = result.get("boq")
        if boq is None:
            st.info("Not available.")
        else:
            if boq["items"]:
                st.dataframe(boq["items"], use_container_width=True)
            else:
                st.info("No BOQ items are supported by the extracted evidence.")
            if boq["assumptions"]:
                st.markdown("**Assumptions:** " + "; ".join(boq["assumptions"]))
            if boq["excluded_items"]:
                st.markdown("**Excluded:** " + "; ".join(boq["excluded_items"]))
            st.download_button(
                "⬇️ Download BOQ JSON",
                data=json.dumps(boq, indent=2, ensure_ascii=False),
                file_name="preliminary_boq.json",
                mime="application/json",
            )

    with tabs[3]:
        st.subheader("Review & Uncertainty")
        review = result.get("review")
        if review is None:
            st.info("Not available.")
        else:
            if review["overall_status"] == "PASS":
                st.success("No issues found. This is still a preliminary takeoff.")
            else:
                st.warning("Review required before relying on these quantities.")
            if review["checks"]:
                st.dataframe(review["checks"], use_container_width=True)
            for w in review["warnings"]:
                st.markdown(f"- ⚠️ {w}")
            if review.get("note"):
                st.caption(review["note"])
