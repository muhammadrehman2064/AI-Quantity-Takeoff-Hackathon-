import os
import json
import streamlit as st

from drawing_processor import process_drawing
from agents import (
    run_drawing_agent,
    run_measurement_agent,
    run_qs_agent,
    run_review_agent,
)
from quantity_engine import build_boq
from rag import load_rag, retrieve_methodology


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Quantity Takeoff",
    page_icon="📐",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📐 AI Architectural Quantity Takeoff")

st.caption(
    "MVP: PDF drawing → drawing analysis → measurement extraction "
    "→ QS takeoff → review → preliminary BOQ"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Settings")

    # Groq API key
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    # --------------------------------------------------------
    # Vision model
    # --------------------------------------------------------

    vision_model = st.selectbox(
        "Vision Model",
        [
            "qwen/qwen3.8-27b",
        ],
        index=0,
        help=(
            "Used for architectural drawing image analysis "
            "and measurement extraction."
        ),
    )

    # --------------------------------------------------------
    # Reasoning / QS model
    # --------------------------------------------------------

    reasoning_model = st.selectbox(
        "QS / Reasoning Model",
        [
            "openai/gpt-oss-120b",
        ],
        index=0,
        help=(
            "Used for quantity takeoff reasoning, preliminary "
            "BOQ preparation and QA/QC review."
        ),
    )

    project_name = st.text_input(
        "Project Name",
        "Sample Residential Project",
    )

    unit_system = st.selectbox(
        "Unit System",
        ["Metric", "Imperial"],
        index=0,
    )

    run_pages = st.slider(
        "Maximum PDF pages",
        1,
        30,
        10,
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload architectural drawing PDF",
    type=["pdf"],
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# RUN PIPELINE
# ============================================================

if uploaded:

    st.info(
        "The MVP will treat extracted measurements as preliminary "
        "takeoff data. Final quantities should be checked against "
        "the original drawings."
    )

    if st.button(
        "🚀 Run AI Quantity Takeoff",
        type="primary",
        use_container_width=True,
    ):

        if not api_key:
            st.error(
                "Enter your Groq API key in the sidebar."
            )
            st.stop()

        try:

            # ------------------------------------------------
            # STEP 1 — PDF PROCESSING
            # ------------------------------------------------

            with st.spinner(
                "Reading PDF and rendering drawing pages..."
            ):

                pages = process_drawing(
                    uploaded.getvalue(),
                    max_pages=run_pages,
                )

            st.write(
                f"Pages prepared for analysis: **{len(pages)}**"
            )

            if not pages:
                st.error(
                    "No drawing pages could be prepared from the PDF."
                )
                st.stop()

            # ------------------------------------------------
            # STEP 2 — DRAWING ANALYSIS
            # Vision model
            # ------------------------------------------------

            with st.spinner(
                "Agent 1/4 — analysing architectural drawing..."
            ):

                drawing_analysis = run_drawing_agent(
                    api_key,
                    vision_model,
                    pages,
                    project_name,
                    unit_system,
                )

            # ------------------------------------------------
            # STEP 3 — MEASUREMENT EXTRACTION
            # Vision model
            # ------------------------------------------------

            with st.spinner(
                "Agent 2/4 — extracting visible measurements..."
            ):

                measurements = run_measurement_agent(
                    api_key,
                    vision_model,
                    pages,
                    drawing_analysis,
                    project_name,
                    unit_system,
                )

            # ------------------------------------------------
            # STEP 4 — RAG METHODOLOGY
            # ------------------------------------------------

            with st.spinner(
                "Loading QS methodology..."
            ):

                rag = load_rag()

                methodology = retrieve_methodology(
                    rag,
                    (
                        "architectural quantity takeoff "
                        "measurement methodology BOQ dimensions "
                        "floor wall door window room area"
                    ),
                    top_k=5,
                )

            # ------------------------------------------------
            # STEP 5 — QS TAKEOFF
            # Reasoning model
            # ------------------------------------------------

            with st.spinner(
                "Agent 3/4 — preparing QS quantities..."
            ):

                qs_result = run_qs_agent(
                    api_key,
                    reasoning_model,
                    drawing_analysis,
                    measurements,
                    methodology,
                    project_name,
                    unit_system,
                )

            # ------------------------------------------------
            # STEP 6 — QA / REVIEW
            # Reasoning model
            # ------------------------------------------------

            with st.spinner(
                "Agent 4/4 — reviewing quantities and uncertainty..."
            ):

                review = run_review_agent(
                    api_key,
                    reasoning_model,
                    drawing_analysis,
                    measurements,
                    qs_result,
                    methodology,
                )

            # ------------------------------------------------
            # STEP 7 — BUILD BOQ
            # ------------------------------------------------

            boq = build_boq(
                qs_result,
                review,
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            st.session_state.result = {
                "project": project_name,
                "vision_model": vision_model,
                "reasoning_model": reasoning_model,
                "drawing_analysis": drawing_analysis,
                "measurements": measurements,
                "qs_result": qs_result,
                "review": review,
                "boq": boq,
            }

            st.success(
                "Takeoff completed successfully."
            )

        except Exception as exc:

            st.error(
                "The AI quantity takeoff pipeline encountered "
                "an error."
            )

            st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result:

    tabs = st.tabs(
        [
            "📋 Summary",
            "📐 Measurements",
            "🧮 BOQ",
            "🔎 Review",
            "🧠 Raw AI Output",
        ]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    with tabs[0]:

        st.subheader("Drawing Analysis")

        st.write(
            f"**Vision Model:** `{result.get('vision_model', '')}`"
        )

        st.write(
            f"**QS / Reasoning Model:** "
            f"`{result.get('reasoning_model', '')}`"
        )

        st.json(
            result["drawing_analysis"]
        )

    # ========================================================
    # MEASUREMENTS
    # ========================================================

    with tabs[1]:

        st.subheader(
            "Extracted Measurements"
        )

        measurements = result["measurements"]

        rows = (
            measurements.get("measurements", [])
            if isinstance(measurements, dict)
            else []
        )

        if rows:

            st.dataframe(
                rows,
                use_container_width=True,
            )

        else:

            st.json(
                measurements
            )

        unverified = (
            measurements.get("unverified_items", [])
            if isinstance(measurements, dict)
            else []
        )

        if unverified:

            st.warning(
                "Some drawing elements could not be verified."
            )

            st.write(
                unverified
            )

    # ========================================================
    # BOQ
    # ========================================================

    with tabs[2]:

        st.subheader(
            "Preliminary BOQ"
        )

        boq = result["boq"]

        rows = boq.get(
            "items",
            []
        )

        if rows:

            st.dataframe(
                rows,
                use_container_width=True,
            )

        else:

            st.json(
                boq
            )

        download = json.dumps(
            boq,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "⬇️ Download BOQ JSON",
            data=download,
            file_name="preliminary_boq.json",
            mime="application/json",
        )

    # ========================================================
    # REVIEW
    # ========================================================

    with tabs[3]:

        st.subheader(
            "Review & Uncertainty"
        )

        st.json(
            result["review"]
        )

    # ========================================================
    # RAW OUTPUT
    # ========================================================

    with tabs[4]:

        st.subheader(
            "Complete Result"
        )

        st.json(
            result
)
