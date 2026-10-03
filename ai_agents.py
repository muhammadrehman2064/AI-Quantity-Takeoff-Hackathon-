import json
import re
import time
from typing import Any, Dict, List

from groq import Groq, BadRequestError, RateLimitError


# ============================================================
# GROQ CLIENT
# ============================================================

def _client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def _extract_json_object(text: str) -> str:
    """
    Extract the first complete JSON object from a model response.

    Handles:
    - <think>...</think>
    - ```json ... ```
    - extra text before/after JSON
    - nested JSON objects
    - braces inside quoted strings
    """

    text = text or ""

    # Remove Qwen reasoning blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    # Find first opening brace
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "The AI response did not contain a JSON object."
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    # JSON appears truncated
    raise ValueError(
        "The AI response contained an incomplete JSON object. "
        "The model response may have been truncated because of the output limit."
    )


def _clean_json(text: str) -> str:
    """
    Clean common JSON formatting mistakes produced by LLMs.
    """

    # Remove BOM / zero-width characters
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")

    # Remove comments occasionally produced by models
    text = re.sub(r"//[^\n\r]*", "", text)

    # Remove trailing commas:
    # {"a": 1,} -> {"a": 1}
    # [1,2,] -> [1,2]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


def _json_from_response(text: str) -> Dict[str, Any]:
    """
    Convert an LLM response into a Python dictionary.

    This is intentionally more tolerant than a direct json.loads()
    because vision/reasoning models occasionally add formatting.
    """

    raw = text or ""

    # First extract the JSON object
    json_text = _extract_json_object(raw)

    # First normal attempt
    try:
        result = json.loads(json_text)

        if not isinstance(result, dict):
            raise ValueError("AI returned JSON, but the root value was not an object.")

        return result

    except json.JSONDecodeError:
        pass

    # Second attempt after cleaning common LLM mistakes
    cleaned = _clean_json(json_text)

    try:
        result = json.loads(cleaned)

        if not isinstance(result, dict):
            raise ValueError("AI returned JSON, but the root value was not an object.")

        return result

    except json.JSONDecodeError as exc:
        # Give a useful diagnostic instead of a cryptic Streamlit crash
        line = exc.lineno
        column = exc.colno

        preview_start = max(0, exc.pos - 250)
        preview_end = min(len(cleaned), exc.pos + 250)

        preview = cleaned[preview_start:preview_end]

        raise ValueError(
            "The AI returned malformed JSON.\n\n"
            f"JSON error: {exc.msg}\n"
            f"Line: {line}\n"
            f"Column: {column}\n\n"
            "Response preview:\n"
            f"{preview}"
        ) from exc


# ============================================================
# TOKEN / RETRY SETTINGS
# ============================================================

# Keep vision output small because free-tier limits can be strict.
VISION_MAX_TOKENS = 1600

# Text agents have more room.
TEXT_MAX_TOKENS = 6000

MAX_RETRIES = 4


# ============================================================
# RETRY HANDLING
# ============================================================

def _retry_wait_seconds(exc) -> float:
    """
    Read:
      try again in 12.3s
      try again in 1m5s

    from Groq rate-limit messages.
    """

    msg = str(exc)

    m = re.search(
        r"try again in (?:(\d+)m)?\s*([\d.]+)s",
        msg,
        flags=re.IGNORECASE,
    )

    if m:
        return (
            float(m.group(1) or 0) * 60
            + float(m.group(2))
            + 1
        )

    return 30.0


def _create(client, **kwargs):
    """
    Groq chat completion with:
    - reasoning_effort='none' attempt
    - fallback if unsupported
    - automatic 429 retry
    """

    last = None

    for attempt in range(MAX_RETRIES):

        try:

            # Try disabling reasoning first.
            try:
                return client.chat.completions.create(
                    reasoning_effort="none",
                    **kwargs,
                )

            except BadRequestError:
                # Some models don't support reasoning_effort.
                return client.chat.completions.create(**kwargs)

        except RateLimitError as exc:

            last = exc

            # A request-size error will not be solved by waiting.
            if "Request too large" in str(exc):
                raise

            wait = min(_retry_wait_seconds(exc), 65)

            time.sleep(wait)

    raise last


# ============================================================
# VISION CALL
# ============================================================

def _vision_call(client, model, prompt, pages):

    prompt = (
        prompt
        + """

IMPORTANT OUTPUT RULES:
- Return ONLY one valid JSON object.
- Do NOT use Markdown.
- Do NOT write ```json.
- Do NOT write explanations before or after JSON.
- Do NOT include <think> tags.
- Keep strings short.
- Maximum 10 items per list.
- Do not invent dimensions.
"""
    )

    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for page in pages:

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

    # Ask Groq for JSON when supported.
    # If the model rejects response_format, _create() will
    # retry without it below.
    try:

        response = _create(
            client,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0,
            max_tokens=VISION_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    except BadRequestError:

        # Some vision/model combinations don't support JSON mode.
        response = _create(
            client,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=0,
            max_tokens=VISION_MAX_TOKENS,
        )

    return response.choices[0].message.content


# ============================================================
# TEXT CALL
# ============================================================

def _text_call(client, model, system, user):

    try:

        response = _create(
            client,
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
            max_tokens=TEXT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

    except BadRequestError:

        response = _create(
            client,
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
            max_tokens=TEXT_MAX_TOKENS,
        )

    return response.choices[0].message.content


# ============================================================
# DRAWING ANALYSIS AGENT
# ============================================================

def run_drawing_agent(
    api_key,
    model,
    pages,
    project_name,
    unit_system,
):

    prompt = f"""
You are the Drawing Analysis Agent for a construction quantity-takeoff application.

Project: {project_name}
Units: {unit_system}

Inspect every supplied drawing page carefully.

Identify only information actually visible in the drawing:

- drawing/page type
- title block information if readable
- scale if explicitly visible
- floor/level
- rooms/spaces
- walls/partitions
- doors and windows
- stairs
- visible dimensions
- symbols and notes
- drawing ambiguities
- unreadable areas

IMPORTANT:

1. Never invent a dimension.
2. If a value cannot be read, mark it as UNKNOWN.
3. Preserve page numbers as evidence.
4. Separate observed facts from assumptions.
5. Return valid JSON only.
6. Keep observations concise.

Return exactly this structure:

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

    result = _vision_call(
        _client(api_key),
        model,
        prompt,
        pages,
    )

    return _json_from_response(result)


# ============================================================
# MEASUREMENT EXTRACTION AGENT
# ============================================================

def run_measurement_agent(
    api_key,
    model,
    pages,
    drawing_analysis,
    project_name,
    unit_system,
):

    prompt = f"""
You are the Measurement Extraction Agent.

Project: {project_name}
Units: {unit_system}

Use the drawing images plus the previous Drawing Agent output.

Extract ONLY measurements/dimensions that are:

1. Clearly visible in the drawing, OR
2. Directly calculable from clearly visible dimensions.

Do NOT guess.

Do NOT use visual estimation as a numeric measurement.

Do NOT convert an uncertain scale into a numeric measurement.

If scale interpretation is uncertain, mark the value UNKNOWN.

For every measurement provide:

- item
- value
- unit
- source_page
- source_dimension_text
- confidence: HIGH/MEDIUM/LOW
- calculation_note

IMPORTANT:
- Keep the list concise.
- Maximum 12 measurements.
- If no reliable measurement exists, return an empty measurements list.
- Return JSON only.

Return exactly this structure:

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

Drawing analysis:

{json.dumps(drawing_analysis, ensure_ascii=False)}
"""

    result = _vision_call(
        _client(api_key),
        model,
        prompt,
        pages,
    )

    return _json_from_response(result)


# ============================================================
# QS AGENT
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

    system = """
You are a senior construction Quantity Surveyor.

Prepare a PRELIMINARY quantity takeoff from evidence supplied by a drawing-analysis pipeline.

Rules:

- Never invent dimensions.
- Never convert UNKNOWN into a numeric quantity.
- Show formulas for calculated quantities.
- Keep source pages.
- Keep confidence.
- If a quantity requires an assumption, explicitly state it.
- Do not claim the BOQ is final.
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

Create a preliminary BOQ.

Include measurable architectural/construction items only where
the evidence supports them.

Do not invent missing dimensions.

Return exactly:

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

    result = _text_call(
        _client(api_key),
        model,
        system,
        user,
    )

    return _json_from_response(result)


# ============================================================
# REVIEW / QA AGENT
# ============================================================

def run_review_agent(
    api_key,
    model,
    drawing_analysis,
    measurements,
    qs_result,
    methodology,
):

    system = """
You are the independent QA/QC reviewer for an AI quantity-takeoff pipeline.

Check:

1. arithmetic/formula consistency
2. unsupported quantities
3. missing source evidence
4. low-confidence measurements
5. double counting
6. unreasonable assumptions
7. whether a quantity should be marked UNABLE_TO_VERIFY

Do not create new dimensions.

Return JSON only.
"""

    user = f"""
DRAWING:

{json.dumps(drawing_analysis, ensure_ascii=False)}

MEASUREMENTS:

{json.dumps(measurements, ensure_ascii=False)}

BOQ:

{json.dumps(qs_result, ensure_ascii=False)}

METHODOLOGY:

{json.dumps(methodology, ensure_ascii=False)}

Return exactly:

{{
  "overall_status": "PASS",
  "checks": [
    {{
      "item_no": null,
      "status": "PASS",
      "issue": "",
      "recommended_action": ""
    }}
  ],
  "critical_warnings": [],
  "review_notes": []
}}

Allowed overall_status values:

PASS
REVIEW_REQUIRED

Allowed check status values:

PASS
REVIEW_REQUIRED
UNABLE_TO_VERIFY
"""

    result = _text_call(
        _client(api_key),
        model,
        system,
        user,
    )

    return _json_from_response(result)
