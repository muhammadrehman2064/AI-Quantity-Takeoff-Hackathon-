import json
from typing import Any, Dict, List

from groq import Groq


# ============================================================
# Groq Client
# ============================================================

def _client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ============================================================
# JSON Helpers
# ============================================================

def _json_from_response(text: str) -> Dict[str, Any]:
    """
    Safely extract JSON from an LLM response.
    Handles normal JSON and ```json fenced responses.
    """

    if not text:
        raise ValueError("AI returned an empty response.")

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Extract JSON object if extra text exists
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "AI response was not valid JSON.\n\n"
            f"Raw response:\n{text[:5000]}"
        ) from exc


# ============================================================
# Vision Call
# ============================================================

def _vision_call(client, model, prompt, pages):
    """
    Sends up to 3 drawing pages to the Groq vision model.

    Groq Qwen 3.8 currently supports a maximum of 3 images
    per request, so the calling functions batch pages.
    """

    if not pages:
        raise ValueError("No drawing pages were supplied.")

    content = [{"type": "text", "text": prompt}]

    for index, page in enumerate(pages, start=1):
        page_number = page.get("page_number", index)

        content.append(
            {
                "type": "text",
                "text": f"--- DRAWING PAGE {page_number} ---",
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{page['mime_type']};base64,"
                        f"{page['image_b64']}"
                    )
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        max_completion_tokens=6000,
    )

    return response.choices[0].message.content


# ============================================================
# Text / Reasoning Call
# ============================================================

def _text_call(client, model, system, user):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        temperature=0,
        max_completion_tokens=6000,
    )

    return response.choices[0].message.content


# ============================================================
# Page Batching
# ============================================================

def _page_batches(pages, batch_size=3):
    """
    Split drawing pages into batches.

    Qwen 3.8 supports max 3 images per request.
    """

    return [
        pages[i:i + batch_size]
        for i in range(0, len(pages), batch_size)
    ]


# ============================================================
# AGENT 1 — DRAWING ANALYSIS
# ============================================================

def run_drawing_agent(
    api_key,
    model,
    pages,
    project_name,
    unit_system,
):
    """
    Analyse architectural drawing pages using vision AI.

    Pages are processed in batches of maximum 3 images.
    """

    client = _client(api_key)

    batches = _page_batches(pages, batch_size=3)

    all_pages = []
    all_global_observations = []
    all_assumptions = []

    for batch_number, batch in enumerate(batches, start=1):

        prompt = f"""
You are the Drawing Analysis Agent for a construction
quantity-takeoff application.

Project: {project_name}
Units: {unit_system}

This is vision analysis batch {batch_number} of {len(batches)}.

Inspect EVERY supplied drawing page carefully.

Identify ONLY information that can actually be observed from
the supplied drawing images.

Look for:

- drawing/page type
- title block information if readable
- scale if explicitly visible
- floor/level
- rooms/spaces
- walls/partitions
- doors and windows
- stairs
- visible dimensions
- symbols
- notes
- grid lines
- drawing references
- drawing ambiguities
- unreadable areas

IMPORTANT RULES:

1. NEVER invent a dimension.
2. NEVER estimate a dimension just because it visually looks like a
   certain size.
3. If a value cannot be confidently read, write UNKNOWN.
4. Preserve the actual drawing page number whenever available.
5. Separate observed facts from assumptions.
6. Do not manufacture scale information.
7. Do not infer hidden geometry.
8. Do not treat text generated by your own reasoning as drawing evidence.
9. Return VALID JSON ONLY.
10. Analyse every supplied page.

Return exactly this JSON structure:

{{
  "drawing_summary": "",
  "pages": [
    {{
      "page_number": 1,
      "drawing_type": "",
      "level": "",
      "scale": "",
      "observations": [],
      "uncertainties": []
    }}
  ],
  "global_observations": [],
  "assumptions": []
}}
"""

        response_text = _vision_call(
            client,
            model,
            prompt,
            batch,
        )

        result = _json_from_response(response_text)

        if isinstance(result.get("pages"), list):
            all_pages.extend(result["pages"])

        if isinstance(result.get("global_observations"), list):
            all_global_observations.extend(
                result["global_observations"]
            )

        if isinstance(result.get("assumptions"), list):
            all_assumptions.extend(
                result["assumptions"]
            )

    return {
        "drawing_summary": (
            f"Drawing analysed in {len(batches)} vision batches."
        ),
        "pages": all_pages,
        "global_observations": all_global_observations,
        "assumptions": all_assumptions,
    }


# ============================================================
# AGENT 2 — MEASUREMENT EXTRACTION
# ============================================================

def run_measurement_agent(
    api_key,
    model,
    pages,
    drawing_analysis,
    project_name,
    unit_system,
):
    """
    Extract visible dimensions and directly calculable measurements.

    Uses the drawing images again because measurement extraction
    requires visual evidence.
    """

    client = _client(api_key)

    batches = _page_batches(pages, batch_size=3)

    all_measurements = []
    all_unverified_items = []

    # Keep the previous drawing analysis available to every batch.
    drawing_analysis_json = json.dumps(
        drawing_analysis,
        ensure_ascii=False,
    )

    for batch_number, batch in enumerate(batches, start=1):

        prompt = f"""
You are the Measurement Extraction Agent for a construction
quantity-takeoff application.

Project: {project_name}
Units: {unit_system}

This is measurement extraction batch {batch_number}
of {len(batches)}.

Use the supplied drawing images plus the previous Drawing
Analysis output.

Your job is to extract ONLY measurements that are:

1. Clearly visible on the drawing, OR
2. Directly calculable from clearly visible dimensions.

DO NOT guess.

DO NOT estimate dimensions based only on visual appearance.

DO NOT convert an UNKNOWN value into a number.

For every measurement provide:

- item
- value
- unit
- source_page
- source_dimension_text if visible
- confidence: HIGH/MEDIUM/LOW
- calculation_note

Examples of acceptable calculations:

Room length × room width = floor area

Wall length × wall height = wall area

Door width × door height = door area

But ONLY when the required dimensions are actually available.

Return JSON ONLY:

{{
  "measurements": [
    {{
      "item": "",
      "value": null,
      "unit": "",
      "source_page": null,
      "source_dimension_text": "",
      "confidence": "HIGH",
      "calculation_note": ""
    }}
  ],
  "unverified_items": []
}}

Previous Drawing Analysis:

{drawing_analysis_json}
"""

        response_text = _vision_call(
            client,
            model,
            prompt,
            batch,
        )

        result = _json_from_response(response_text)

        if isinstance(result.get("measurements"), list):
            all_measurements.extend(
                result["measurements"]
            )

        if isinstance(result.get("unverified_items"), list):
            all_unverified_items.extend(
                result["unverified_items"]
            )

    return {
        "measurements": all_measurements,
        "unverified_items": all_unverified_items,
    }


# ============================================================
# AGENT 3 — QS TAKEOFF
# ============================================================

def run_qs_agent(
    api_key,
    model,
    drawing_analysis,
    measurements,
    methodology,
    project_name,
    unit_system,
):
    """
    Text/reasoning agent.

    This agent does NOT inspect images.
    It works only from evidence extracted by the vision agents.
    """

    system = """
You are a senior construction Quantity Surveyor.

Prepare a PRELIMINARY quantity takeoff from evidence supplied
by a drawing-analysis pipeline.

STRICT EVIDENCE RULES:

- Never invent dimensions.
- Never convert UNKNOWN into a numeric quantity.
- Never create missing dimensions.
- Show formulas for calculated quantities.
- Keep source page references.
- Keep confidence levels.
- If a quantity requires an assumption, explicitly state it.
- Do not claim that the BOQ is final.
- Do not silently assume wall heights, slab thicknesses,
  openings, wastage, or construction details.
- If evidence is insufficient, exclude the quantity or mark
  it as requiring verification.
- Return JSON only.
"""

    user = f"""
Project: {project_name}
Units: {unit_system}

DRAWING ANALYSIS:
{json.dumps(drawing_analysis, ensure_ascii=False)}

MEASUREMENTS:
{json.dumps(measurements, ensure_ascii=False)}

QS METHODOLOGY / RAG CONTEXT:
{json.dumps(methodology, ensure_ascii=False)}

Prepare a preliminary BOQ.

Include measurable architectural/construction items ONLY where
the supplied evidence supports the quantity.

JSON schema:

{{
  "items": [
    {{
      "item_no": 1,
      "description": "",
      "unit": "m2",
      "quantity": null,
      "formula": "",
      "source_pages": [],
      "confidence": "HIGH",
      "basis": "",
      "assumption": ""
    }}
  ],
  "assumptions": [],
  "excluded_items": []
}}
"""

    return _json_from_response(
        _text_call(
            _client(api_key),
            model,
            system,
            user,
        )
    )


# ============================================================
# AGENT 4 — REVIEW / QA
# ============================================================

def run_review_agent(
    api_key,
    model,
    drawing_analysis,
    measurements,
    qs_result,
    methodology,
):
    """
    Independent text-based QA/QC review.
    """

    system = """
You are the independent QA/QC reviewer for an AI quantity-
takeoff pipeline.

Review the supplied evidence and preliminary BOQ.

Check:

1. arithmetic/formula consistency
2. unsupported quantities
3. missing source evidence
4. low-confidence measurements
5. double counting
6. unreasonable assumptions
7. quantities that should be marked UNABLE_TO_VERIFY
8. whether the BOQ follows the supplied methodology

STRICT RULES:

- Do not create new dimensions.
- Do not repair missing evidence by guessing.
- Do not introduce new quantities.
- If evidence is insufficient, flag it.
- Return JSON only.
"""

    user = f"""
DRAWING ANALYSIS:

{json.dumps(drawing_analysis, ensure_ascii=False)}

MEASUREMENTS:

{json.dumps(measurements, ensure_ascii=False)}

BOQ:

{json.dumps(qs_result, ensure_ascii=False)}

METHODOLOGY:

{json.dumps(methodology, ensure_ascii=False)}

Return exactly:

{{
  "overall_status": "PASS|REVIEW_REQUIRED",
  "checks": [
    {{
      "item_no": null,
      "status": "PASS|REVIEW_REQUIRED|UNABLE_TO_VERIFY",
      "issue": "",
      "recommended_action": ""
    }}
  ],
  "critical_warnings": [],
  "review_notes": []
}}
"""

    return _json_from_response(
        _text_call(
            _client(api_key),
            model,
            system,
            user,
        )
    )
