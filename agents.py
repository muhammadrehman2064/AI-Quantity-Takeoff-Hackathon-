import json
import re
from typing import Any, Dict, List

from groq import Groq


def _client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _json_from_response(text: str) -> Dict[str, Any]:
    text = text or ""
    # Qwen reasoning models can emit <think>...</think> before the answer
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _vision_call(client, model, prompt, pages):
    content = [{"type": "text", "text": prompt}]
    for page in pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{page['mime_type']};base64,{page['image_b64']}"
                },
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=8000,
    )
    return response.choices[0].message.content


def _text_call(client, model, system, user):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=8000,
    )
    return response.choices[0].message.content


def run_drawing_agent(api_key, model, pages, project_name, unit_system):
    prompt = f"""
You are the Drawing Analysis Agent for a construction quantity-takeoff application.

Project: {project_name}
Units: {unit_system}

Inspect every supplied drawing page carefully. Identify:
- drawing/page type
- title block information if readable
- scale if explicitly visible
- floor/level
- rooms/spaces
- walls/partitions
- doors and windows
- stairs
- dimensions that are actually visible
- symbols and notes
- drawing ambiguities or unreadable areas

IMPORTANT:
1. Never invent a dimension.
2. If a value cannot be read, mark it as UNKNOWN.
3. Preserve page numbers as evidence.
4. Separate observed facts from assumptions.
5. Return valid JSON only.

Schema:
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
    return _json_from_response(_vision_call(_client(api_key), model, prompt, pages))


def run_measurement_agent(api_key, model, pages, drawing_analysis, project_name, unit_system):
    prompt = f"""
You are the Measurement Extraction Agent.

Project: {project_name}
Units: {unit_system}

Use the drawing images plus the previous Drawing Agent output.

Extract ONLY measurements/dimensions that are visible or directly calculable from visible dimensions.
Do not guess scale-based measurements unless the drawing explicitly provides a reliable scale and the image quality supports using it. If scale interpretation is uncertain, mark it UNKNOWN.

For every measurement provide:
- item
- value
- unit
- source_page
- source_dimension_text if visible
- confidence: HIGH/MEDIUM/LOW
- calculation_note

Return JSON only:
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
    return _json_from_response(_vision_call(_client(api_key), model, prompt, pages))


def run_qs_agent(api_key, model, drawing_analysis, measurements, methodology, project_name, unit_system):
    system = """
You are a senior construction Quantity Surveyor. Prepare a preliminary quantity takeoff
from evidence supplied by a drawing-analysis pipeline.

Rules:
- Never invent dimensions.
- Never convert an UNKNOWN value into a numeric quantity.
- Show formulas for calculated quantities.
- Keep source page and confidence.
- If a quantity requires an assumption, put the assumption explicitly in the assumptions field.
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

Create a preliminary BOQ. Include measurable architectural/construction items only where
the evidence supports them.

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
    return _json_from_response(_text_call(_client(api_key), model, system, user))


def run_review_agent(api_key, model, drawing_analysis, measurements, qs_result, methodology):
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

Do not create new dimensions. Return JSON only.
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

JSON:
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
    return _json_from_response(_text_call(_client(api_key), model, system, user))
