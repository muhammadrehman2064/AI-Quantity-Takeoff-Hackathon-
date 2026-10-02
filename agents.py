import json
from typing import Any, Dict

from groq import Groq


# ============================================================
# GROQ CLIENT
# ============================================================

def _client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ============================================================
# JSON PARSER
# ============================================================

def _json_from_response(text: str) -> Dict[str, Any]:
    """
    Convert an AI response into a Python dictionary.

    Handles:
    - normal JSON
    - ```json ... ```
    - responses containing extra text around JSON
    """

    if not text:
        raise ValueError("AI returned an empty response.")

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Extract JSON object from surrounding text
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        result = json.loads(text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "AI response was not valid JSON.\n\n"
            "Raw AI response:\n"
            f"{text[:5000]}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("AI response JSON is not an object.")

    return result


# ============================================================
# PAGE BATCHING
# ============================================================

def _page_batches(pages, batch_size=3):
    """
    Split PDF pages into batches.

    Qwen vision requests are limited to a small number of
    images, therefore we keep maximum 3 pages per request.
    """

    return [
        pages[i:i + batch_size]
        for i in range(0, len(pages), batch_size)
    ]


# ============================================================
# VISION CALL
# ============================================================

def _vision_call(client, model, prompt, pages):
    """
    Send architectural drawing images to the vision model.

    IMPORTANT:
    max_tokens is intentionally kept below the current
    Free-plan OTPM limit reported by Groq.
    """

    if not pages:
        raise ValueError("No drawing pages supplied.")

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for index, page in enumerate(pages, start=1):

        page_number = page.get("page_number", index)

        content.append(
            {
                "type": "text",
                "text": f"DRAWING PAGE {page_number}",
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
        max_tokens=700,
    )

    return response.choices[0].message.content


# ============================================================
# TEXT / REASONING CALL
# ============================================================

def _text_call(client, model, system, user):
    """
    Text-only call for QS reasoning and QA review.

    Uses GPT-OSS 120B in the current application.
    """

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
        max_tokens=1200,
    )

    return response.choices[0].message.content


# ============================================================
# AGENT 1
# DRAWING ANALYSIS
# ============================================================

def run_drawing_agent(
    api_key,
    model,
    pages,
    project_name,
    unit_system,
):
    """
    Analyse architectural drawing pages.

    Vision model:
        qwen/qwen3.8-27b

    Output is deliberately compact because the Free Groq plan
    has a strict output-token-per-minute limit.
    """

    client = _client(api_key)

    batches = _page_batches(
        pages,
        batch_size=3,
    )

    all_pages = []
    all_global_observations = []
    all_assumptions = []

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        prompt = f"""
You are an architectural drawing analysis AI.

Project: {project_name}
Units: {unit_system}

Analyse ONLY the supplied drawing images.

Extract visible information about:
- drawing type
- title/block information if readable
- floor/level
- scale if explicitly printed
- rooms/spaces
- walls/partitions
- doors
- windows
- stairs
- visible dimensions
- notes/symbols
- drawing uncertainties

STRICT RULES:

1. Never invent dimensions.
2. Never estimate dimensions from appearance.
3. If information is unreadable, write UNKNOWN.
4. Use the actual drawing page number.
5. Do not assume a scale unless explicitly shown.
6. Keep observations short.
7. Return JSON ONLY.
8. Do not write explanations outside JSON.

Return:

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

        result = _json_from_response(
            response_text
        )

        pages_result = result.get(
            "pages",
            [],
        )

        if isinstance(pages_result, list):
            all_pages.extend(
                pages_result
            )

        observations = result.get(
            "global_observations",
            [],
        )

        if isinstance(observations, list):
            all_global_observations.extend(
                observations
            )

        assumptions = result.get(
            "assumptions",
            [],
        )

        if isinstance(assumptions, list):
            all_assumptions.extend(
                assumptions
            )

    return {
        "drawing_summary": (
            f"Architectural drawing analysed "
            f"in {len(batches)} vision batches."
        ),
        "pages": all_pages,
        "global_observations": all_global_observations,
        "assumptions": all_assumptions,
    }


# ============================================================
# AGENT 2
# MEASUREMENT EXTRACTION
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
    Extract visible dimensions and directly calculable
    measurements from architectural drawings.

    Vision model:
        qwen/qwen3.8-27b
    """

    client = _client(api_key)

    batches = _page_batches(
        pages,
        batch_size=3,
    )

    all_measurements = []
    all_unverified_items = []

    drawing_analysis_json = json.dumps(
        drawing_analysis,
        ensure_ascii=False,
    )

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):

        prompt = f"""
You are a construction measurement extraction AI.

Project: {project_name}
Units: {unit_system}

Analyse the supplied architectural drawing images.

Extract ONLY:

1. Dimensions visibly written on drawings.
2. Measurements directly calculable from visible dimensions.

Examples:
- room length
- room width
- room area
- wall length
- door width/height
- window width/height
- stair dimensions
- floor dimensions

STRICT RULES:

- Never guess.
- Never estimate from visual appearance.
- Never invent dimensions.
- Never use an assumed scale.
- UNKNOWN values must remain UNKNOWN.
- Every measurement needs a source page.
- Preserve the original dimension text when readable.
- Keep calculations simple.
- Return JSON ONLY.
- Keep the response concise.

Confidence must be:
HIGH
MEDIUM
LOW

Return:

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

Previous drawing analysis:

{drawing_analysis_json}
"""

        response_text = _vision_call(
            client,
            model,
            prompt,
            batch,
        )

        result = _json_from_response(
            response_text
        )

        measurements = result.get(
            "measurements",
            [],
        )

        if isinstance(measurements, list):
            all_measurements.extend(
                measurements
            )

        unverified = result.get(
            "unverified_items",
            [],
        )

        if isinstance(unverified, list):
            all_unverified_items.extend(
                unverified
            )

    return {
        "measurements": all_measurements,
        "unverified_items": all_unverified_items,
    }


# ============================================================
# AGENT 3
# QS QUANTITY TAKEOFF
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
    Prepare preliminary BOQ from extracted evidence.

    Text model:
        openai/gpt-oss-120b
    """

    system = """
You are a senior construction Quantity Surveyor.

Prepare a PRELIMINARY quantity takeoff using only the
evidence provided by the drawing-analysis pipeline.

STRICT RULES:

1. Never invent dimensions.
2. Never turn UNKNOWN into a number.
3. Never create missing dimensions.
4. Show formulas.
5. Keep source pages.
6. Keep confidence levels.
7. State assumptions explicitly.
8. Do not double count.
9. Do not claim quantities are final.
10. Exclude quantities without sufficient evidence.
11. Return JSON ONLY.
12. Keep output concise.
"""

    user = f"""
Project: {project_name}
Units: {unit_system}

DRAWING ANALYSIS:
{json.dumps(drawing_analysis, ensure_ascii=False)}

MEASUREMENTS:
{json.dumps(measurements, ensure_ascii=False)}

QS METHODOLOGY:
{json.dumps(methodology, ensure_ascii=False)}

Prepare a preliminary BOQ.

Only include items for which the supplied evidence supports
a quantity.

Use:

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

Return JSON ONLY.
"""

    response_text = _text_call(
        _client(api_key),
        model,
        system,
        user,
    )

    return _json_from_response(
        response_text
    )


# ============================================================
# AGENT 4
# QA / QC REVIEW
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
    Independent QA/QC review.

    Text model:
        openai/gpt-oss-120b
    """

    system = """
You are an independent QA/QC reviewer for an AI construction
quantity-takeoff system.

Review the preliminary BOQ against the supplied evidence.

Check:

1. arithmetic
2. formulas
3. unsupported quantities
4. missing source pages
5. low-confidence measurements
6. double counting
7. assumptions
8. quantities requiring verification

STRICT RULES:

- Do not create new dimensions.
- Do not create new quantities.
- Do not repair missing evidence by guessing.
- Flag unsupported quantities.
- Return JSON ONLY.
- Keep the response concise.
"""

    user = f"""
DRAWING ANALYSIS:

{json.dumps(drawing_analysis, ensure_ascii=False)}

MEASUREMENTS:

{json.dumps(measurements, ensure_ascii=False)}

PRELIMINARY BOQ:

{json.dumps(qs_result, ensure_ascii=False)}

METHODOLOGY:

{json.dumps(methodology, ensure_ascii=False)}

Return:

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

Return JSON ONLY.
"""

    response_text = _text_call(
        _client(api_key),
        model,
        system,
        user,
    )

    return _json_from_response(
        response_text
        )
