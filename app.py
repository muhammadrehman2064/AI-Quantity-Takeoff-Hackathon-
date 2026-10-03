import os
import json
import streamlit as st

from drawing_processor import process_drawing
from ai_agents import run_drawing_agent, run_measurement_agent, run_qs_agent, run_review_agent
from quantity_engine import build_boq
from rag import load_rag, retrieve_methodology

st.set_page_config(page_title="AI Quantity Takeoff", page_icon="📐", layout="wide")

st.title("📐 AI Architectural Quantity Takeoff")
st.caption("MVP: PDF drawing → drawing analysis → measurement extraction → QS takeoff → review → preliminary BOQ")

with st.sidebar:
    st.header("Project Settings")
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    vision_choice = st.selectbox(
        "Vision model (Agents 1 & 2)",
        ["qwen/qwen3.8-27b", "Custom..."],
        index=0,
    )
    vision_model = (
        st.text_input("Custom vision model ID", "") if vision_choice == "Custom..." else vision_choice
    )
    text_choice = st.selectbox(
        "Text model (Agents 3 & 4)",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "Custom..."],
        index=0,
    )
    text_model = (
        st.text_input("Custom text model ID", "") if text_choice == "Custom..." else text_choice
    )
    project_name = st.text_input("Project Name", "Sample Residential Project")
    unit_system = st.selectbox("Unit System", ["Metric", "Imperial"], index=0)
    run_pages = st.slider("Maximum PDF pages", 1, 5, 3)

uploaded = st.file_uploader("Upload architectural drawing PDF", type=["pdf"])

if "result" not in st.session_state:
    st.session_state.result = None

if uploaded:
    st.info("The MVP will treat extracted measurements as preliminary takeoff data. Final quantities should be checked against the original drawings.")

    if st.button("🚀 Run AI Quantity Takeoff", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
            st.stop()

        try:
            with st.spinner("Reading PDF and rendering drawing pages..."):
                pages = process_drawing(uploaded.getvalue(), max_pages=run_pages)

            st.write(f"Pages prepared for analysis: **{len(pages)}**")

            with st.spinner("Agent 1/4 — analysing drawing..."):
                drawing_analysis = run_drawing_agent(api_key, vision_model, pages, project_name, unit_system)

            with st.spinner("Agent 2/4 — extracting measurements..."):
                measurements = run_measurement_agent(
                    api_key, vision_model, pages, drawing_analysis, project_name, unit_system
                )

            rag = load_rag()
            methodology = retrieve_methodology(
                rag,
                "architectural quantity takeoff measurement methodology BOQ dimensions floor wall door window room area",
                top_k=5,
            )

            with st.spinner("Agent 3/4 — preparing QS quantities..."):
                qs_result = run_qs_agent(
                    api_key,
                    text_model,
                    drawing_analysis,
                    measurements,
                    methodology,
                    project_name,
                    unit_system,
                )

            with st.spinner("Agent 4/4 — reviewing quantities and uncertainty..."):
                review = run_review_agent(
                    api_key, text_model, drawing_analysis, measurements, qs_result, methodology
                )

            boq = build_boq(qs_result, review)

            st.session_state.result = {
                "project": project_name,
                "drawing_analysis": drawing_analysis,
                "measurements": measurements,
                "qs_result": qs_result,
                "review": review,
                "boq": boq,
            }
            st.success("Takeoff completed.")

        except Exception as exc:
            if type(exc).__name__ == "NotFoundError":
                st.error(
                    "Selected Groq model is not available (it may have been retired). "
                    "Check https://console.groq.com/docs/models and enter a valid ID via 'Custom...'."
                )
            st.exception(exc)

result = st.session_state.result

if result:
    tabs = st.tabs(["📋 Summary", "📐 Measurements", "🧮 BOQ", "🔎 Review", "🧠 Raw AI Output"])

    with tabs[0]:
        st.subheader("Drawing Analysis")
        st.json(result["drawing_analysis"])

    with tabs[1]:
        st.subheader("Extracted Measurements")
        measurements = result["measurements"]
        rows = measurements.get("measurements", []) if isinstance(measurements, dict) else []
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.json(measurements)

    with tabs[2]:
        st.subheader("Preliminary BOQ")
        boq = result["boq"]
        rows = boq.get("items", [])
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.json(boq)

        download = json.dumps(boq, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Download BOQ JSON",
            data=download,
            file_name="preliminary_boq.json",
            mime="application/json",
        )

    with tabs[3]:
        st.subheader("Review & Uncertainty")
        st.json(result["review"])

    with tabs[4]:
        st.subheader("Complete Result")
        st.json(result)
