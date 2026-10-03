"""Groq-backed agents for the quantity-takeoff MVP.

Pipeline (4 AI calls, all with tiny JSON outputs):
    1. run_drawing_agent      (vision)  drawing pages  -> compact drawing analysis + visible dimensions
    2. run_measurement_agent  (text)    dimensions     -> numeric measurements
    3. run_qs_agent           (text)    measurements   -> preliminary BOQ items
    4. run_review_agent       (text)    BOQ            -> compact list of issues

Design rules:
    * Output budgets stay far below Groq's ~1000 output-tokens-per-minute limit for qwen models.
    * Output-token usage is paced per model so consecutive calls do not trigger 429s.
    * 429s are classified: oversized / daily limits fail immediately, per-minute limits are
      retried at most twice, and the SDK's own hidden retries are disabled.
    * Every parse failure becomes an AgentError with a user-friendly message (no tracebacks).
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)


class AgentError(Exception):
    """User-facing error. `str(exc)` is safe to show directly; `detail` is optional debug text."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


# ============================================================
# TOKEN BUDGETS
# ============================================================

# Hard limit enforced by Groq for qwen models on this account (output tokens per minute).
QWEN_OTPM_LIMIT = 1000

# max_tokens per agent. All stay below the 1000 OTPM limit (never request more than the limit).
VISION_MAX_TOKENS = 800
MEASURE_MAX_TOKENS = 700
QS_MAX_TOKENS = 850
REVIEW_MAX_TOKENS = 500

_BASE_BUDGET = {
    "drawing": VISION_MAX_TOKENS,
    "measure": MEASURE_MAX_TOKENS,
    "qs": QS_MAX_TOKENS,
    "review": REVIEW_MAX_TOKENS,
}

MAX_RATE_RETRIES = 2      # only for temporary per-minute limits
MAX_RATE_WAIT_S = 75.0    # never sleep longer than this for a single retry


def _otpm_limit(model: str) -> Optional[int]:
    """Known hard output-token-per-minute limit for a model, or None if unknown."""
    return QWEN_OTPM_LIMIT if "qwen" in (model or "").lower() else None


def _max_tokens(model: str, kind: str) -> int:
    base = _BASE_BUDGET[kind]
    limit = _otpm_limit(model)
    if limit:
        return min(base, int(limit * 0.85))  # 850 for qwen: always under the limit
    if "gpt-oss" in (model or "").lower():
        return base * 2  # reasoning tokens count toward max_tokens on gpt-oss
    return base


def _reasoning_effort(model: str) -> Optional[str]:
    m = (model or "").lower()
    if "gpt-oss" in m:
        return "low"
    if "qwen" in m:
        return "none"
    return None


# ============================================================
# OUTPUT-TOKEN PACING (per model, rolling 60 s window)
# ============================================================

_OUTPUT_LOG: Dict[str, List[Tuple[float, int]]] = {}
_UNSUPPORTED: Dict[str, set] = {}  # per-model request params the API rejected


def _wait_for_budget(model: str, requested: int) -> None:
    limit = _otpm_limit(model)
    if not limit:
        return
    log = _OUTPUT_LOG.setdefault(model, [])
    while True:
        now = time.time()
        log[:] = [(t, n) for (t, n) in log if now - t < 60.0]
        used = sum(n for _, n in log)
        if used + requested <= limit or not log:
            return
        time.sleep(max(0.5, log[0][0] + 60.0 - now + 0.5))


def _record_usage(model: str, tokens: int) -> None:
    if _otpm_limit(model):
        _OUTPUT_LOG.setdefault(model, []).append((time.time(), max(0, int(tokens))))


# ============================================================
# GROQ ERRORS
# ============================================================

def _client(api_key: str) -> Groq:
    # max_retries=0: the SDK would otherwise silently retry 429s and burn the token budget.
    return Groq(api_key=api_key, max_retries=0, timeout=120.0)


def _classify_429(exc) -> Tuple[str, float, str]:
    """Return (kind, wait_seconds, message). kind: 'too_large' | 'daily' | 'temporary'."""
    msg = str(exc)
    low = msg.lower()

    limit_m = re.search(r"limit\s+(\d+)", msg, flags=re.IGNORECASE)
    req_m = re.search(r"requested\s+(\d+)", msg, flags=re.IGNORECASE)
    if limit_m and req_m and int(req_m.group(1)) > int(limit_m.group(1)):
        return (
            "too_large",
            0.0,
            f"Groq refused the request: it needs {req_m.group(1)} tokens but the model allows "
            f"only {limit_m.group(1)} per minute. Choose a model with a higher limit.",
        )
    if "request too large" in low or "reduce your message size" in low:
        return (
            "too_large",
            0.0,
            "The request is too large for this Groq model. Use fewer or smaller PDF pages.",
        )
    if "per day" in low or "(tpd)" in low or "(rpd)" in low:
        return (
            "daily",
            0.0,
            "Groq daily quota reached for this model. Try again later or choose another model.",
        )

    wait = 20.0
    w = re.search(r"try again in (?:(\d+)m)?\s*(?:([\d.]+)s)?", msg, flags=re.IGNORECASE)
    if w and (w.group(1) or w.group(2)):
        wait = float(w.group(1) or 0) * 60 + float(w.group(2) or 0) + 1
    return "temporary", wait, "Groq per-minute rate limit reached. Wait about a minute and run again."


def _map_error(exc: Exception) -> AgentError:
    if isinstance(exc, AuthenticationError):
        return AgentError("Invalid Groq API key. Check the key in the sidebar.")
    if isinstance(exc, NotFoundError):
        return AgentError(
            "The selected Groq model is not available (it may be retired). "
            "Check https://console.groq.com/docs/models and use 'Custom...' to enter a valid ID."
        )
    if isinstance(exc, PermissionDeniedError):
        return AgentError(
            "Groq denied access to this model. Enable it in the Groq console (Settings → Limits/Models)."
        )
    if isinstance(exc, APIConnectionError):
        return AgentError("Could not reach the Groq API. Check the network connection and try again.")
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", "?")
        if status == 413:
            return AgentError("The drawing images are too large for Groq. Use fewer PDF pages.")
        return AgentError(f"Groq API error (HTTP {status}).", detail=str(exc)[:500])
    return AgentError(f"Unexpected error while calling Groq: {type(exc).__name__}", detail=str(exc)[:500])


def _failed_generation(exc) -> str:
    """Groq's JSON mode returns HTTP 400 json_validate_failed with the partial text attached."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            return str(err.get("failed_generation") or "")
    return ""


# ============================================================
# CHAT CALL
# ============================================================

def _chat(client, model: str, messages: list, max_tokens: int) -> Tuple[str, str]:
    """One chat completion. Returns (text, finish_reason). Raises AgentError only."""
    unsupported = _UNSUPPORTED.setdefault(model, set())
    effort = _reasoning_effort(model)
    rate_retries = 0

    while True:
        kwargs: Dict[str, Any] = dict(
            model=model, messages=messages, temperature=0, max_tokens=max_tokens
        )
        if "response_format" not in unsupported:
            kwargs["response_format"] = {"type": "json_object"}
        if effort and "reasoning_effort" not in unsupported:
            kwargs["reasoning_effort"] = effort

        _wait_for_budget(model, max_tokens)

        try:
            resp = client.chat.completions.create(**kwargs)

        except BadRequestError as exc:
            low = str(exc).lower()
            partial = _failed_generation(exc)
            if "json_validate_failed" in low or partial:
                # JSON mode produced invalid/truncated JSON: hand the partial text to the parser.
                _record_usage(model, max_tokens)
                return partial, "length"
            if "response_format" in kwargs and (
                "response_format" in low or "json_object" in low or "json mode" in low
            ):
                unsupported.add("response_format")
                continue
            if "reasoning_effort" in kwargs and "reasoning" in low:
                unsupported.add("reasoning_effort")
                continue
            raise AgentError("Groq rejected the request.", detail=str(exc)[:500]) from exc

        except RateLimitError as exc:
            kind, wait, message = _classify_429(exc)
            if kind != "temporary" or rate_retries >= MAX_RATE_RETRIES or wait > MAX_RATE_WAIT_S:
                raise AgentError(message, detail=str(exc)[:500]) from exc
            rate_retries += 1
            time.sleep(wait)
            continue

        except AgentError:
            raise
        except Exception as exc:  # network, auth, 404, 5xx ...
            raise _map_error(exc) from exc

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        used = getattr(usage, "completion_tokens", None) or max_tokens
        _record_usage(model, used)
        return (choice.message.content or ""), (choice.finish_reason or "")


# ============================================================
# ROBUST JSON PARSING
# ============================================================

def _strip_noise(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Unclosed <think> (output cut off while reasoning): drop everything from it.
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return text.strip()


def _find_object(text: str) -> Tuple[str, bool]:
    """Return (candidate, complete). `complete` is False when the object is cut off."""
    start = text.find("{")
    if start == -1:
        return "", False
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], True
    return text[start:], False


def _salvage_truncated(s: str) -> Optional[str]:
    """Cut a truncated JSON object back to its last complete value and close open brackets."""
    stack: List[str] = []
    in_str, esc = False, False
    safe: Optional[Tuple[int, str]] = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = (i + 1, "".join(reversed(stack)))
        elif ch == "," and stack and (stack[-1] == "]" or len(stack) == 1):
            # Only cut between array elements (or root fields) so half-written objects are dropped.
            safe = (i, "".join(reversed(stack)))
    if safe is None:
        return None
    return s[:safe[0]] + safe[1]


def _loads(text: str) -> Optional[dict]:
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            value = json.loads(candidate, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_json_response(text: str, finish_reason: str = "") -> Dict[str, Any]:
    """Convert a model reply to a dict. Tolerates fences, <think>, extra text, trailing commas
    and truncated output. Raises AgentError (never a raw JSON exception)."""
    cleaned = _strip_noise(text)
    candidate, complete = _find_object(cleaned)

    if candidate:
        result = _loads(candidate)
        if result is not None:
            return result
        if not complete:
            repaired = _salvage_truncated(candidate)
            result = _loads(repaired) if repaired else None
            if result is not None:
                result["_warning"] = "AI output was cut off by the token limit; partial result recovered."
                return result

    if finish_reason == "length" or (candidate and not complete):
        raise AgentError(
            "The AI response was cut off by the output-token limit before it finished. "
            "Run again, use fewer PDF pages, or choose a different model."
        )
    if not candidate:
        raise AgentError(
            "The AI did not return any JSON (it may have spent its output on reasoning). "
            "Run again, or choose a different model."
        )
    raise AgentError(
        "The AI returned malformed JSON. Run again, or choose a different model.",
        detail=cleaned[:400],
    )


# ============================================================
# NORMALISATION HELPERS (keep schemas stable between agents)
# ============================================================

_CONF = {"HIGH", "MEDIUM", "LOW"}


def _text(x: Any, limit: int = 160) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        x = ", ".join(str(v) for v in x.values() if v not in (None, ""))
    return str(x).strip()[:limit]


def _strs(v: Any, n: int, limit: int = 120) -> List[str]:
    if v is None or v == "":
        return []
    if not isinstance(v, list):
        v = [v]
    out = [_text(x, limit) for x in v]
    return [s for s in out if s][:n]


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "")
    if re.fullmatch(r"-?\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _int(v: Any) -> Optional[int]:
    n = _num(v)
    return int(n) if n is not None else None


def _conf(v: Any) -> str:
    c = str(v or "").strip().upper()
    return c if c in _CONF else "LOW"


def _pages_list(v: Any) -> List[int]:
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    out = [_int(x) for x in v]
    return [p for p in out if p is not None]


def _carry_warning(src: dict, dst: dict) -> dict:
    if src.get("_warning"):
        dst["_warning"] = src["_warning"]
    return dst


def _rows(v: Any) -> List[dict]:
    return [r for r in v if isinstance(r, dict)] if isinstance(v, list) else []


# ============================================================
# SHARED PROMPT RULES
# ============================================================

JSON_RULES = (
    "OUTPUT RULES: return ONE JSON object only. No explanations, no reasoning, no Markdown, "
    "no <think> blocks. Short strings (max 8 words). Only the schema fields. "
    "Do not repeat information. Never invent dimensions or quantities; use null or UNKNOWN "
    "when not reliable."
)


def _run(client, model, messages, kind) -> Dict[str, Any]:
    text, finish = _chat(client, model, messages, _max_tokens(model, kind))
    return parse_json_response(text, finish)


# ============================================================
# AGENT 1: DRAWING ANALYSIS (vision, single call)
# ============================================================

def _hint_block(pages: List[Dict[str, Any]]) -> str:
    lines = []
    for p in pages:
        hints = p.get("text_hints") or []
        if hints:
            lines.append(f"p{p['page_number']}: " + " | ".join(hints))
    return "\n".join(lines) or "none"


def run_drawing_agent(api_key, model, pages, project_name, unit_system) -> Dict[str, Any]:
    page_ids = ", ".join(str(p["page_number"]) for p in pages)
    prompt = f"""Architectural drawing reader for quantity takeoff.
Project: {project_name}. Units: {unit_system}.
Images are drawing pages {page_ids} in that order.
PDF text-layer hints (may be noisy or empty):
{_hint_block(pages)}

Record only what is visible on each page: drawing type, level, printed scale, visible dimensions
(e.g. "Living 4.20x3.60 m", "Wall 5200 mm"), and counted elements (e.g. "5 doors", "3 windows",
"stair"). Per page max 8 dimensions, 8 elements, 3 uncertainties. If a value is unreadable write UNKNOWN.

{JSON_RULES}

Schema:
{{"summary":"","pages":[{{"page":1,"type":"","level":"","scale":"","dimensions":[],"elements":[],"uncertainties":[]}}]}}"""

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{p['mime_type']};base64,{p['image_b64']}"},
            }
        )

    raw = _run(_client(api_key), model, [{"role": "user", "content": content}], "drawing")

    out_pages = []
    for i, p in enumerate(_rows(raw.get("pages"))):
        out_pages.append(
            {
                "page": _int(p.get("page", p.get("page_number"))) or (pages[i]["page_number"] if i < len(pages) else i + 1),
                "type": _text(p.get("type", p.get("drawing_type")), 60),
                "level": _text(p.get("level"), 40),
                "scale": _text(p.get("scale"), 40),
                "dimensions": _strs(p.get("dimensions"), 10),
                "elements": _strs(p.get("elements"), 10),
                "uncertainties": _strs(p.get("uncertainties"), 4),
            }
        )
    result = {"summary": _text(raw.get("summary", raw.get("drawing_summary")), 300), "pages": out_pages}
    return _carry_warning(raw, result)


# ============================================================
# AGENT 2: MEASUREMENTS (text only: reuses the drawing agent's dimensions, no images)
# ============================================================

def run_measurement_agent(api_key, model, drawing_analysis, project_name, unit_system) -> Dict[str, Any]:
    compact = {
        "pages": [
            {k: p.get(k) for k in ("page", "scale", "dimensions", "elements")}
            for p in drawing_analysis.get("pages", [])
        ]
    }
    unit_hint = "m, m2, nr" if unit_system == "Metric" else "ft, ft2, nr"
    system = "You convert drawing dimensions into numeric measurements for a quantity surveyor. " + JSON_RULES
    user = f"""Project: {project_name}. Units: {unit_system}.

DRAWING DATA:
{json.dumps(compact, ensure_ascii=False)}

Rules:
- One measurement per clearly stated dimension or counted element (use unit "nr" for counts).
- Normalise units to {unit_hint} (e.g. 3600 mm = 3.6 m).
- You may compute an area/length ONLY from stated dimensions (e.g. 4.2 x 3.6 = 15.12 m2); confidence MEDIUM.
  Stated directly = HIGH. Unclear = LOW.
- Do not use scale to estimate sizes. Anything unreadable or UNKNOWN goes in "unverified_items".
- Max 15 measurements.

Schema:
{{"measurements":[{{"item":"","value":null,"unit":"","page":null,"confidence":"HIGH"}}],"unverified_items":[]}}"""

    raw = _run(
        _client(api_key),
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "measure",
    )

    measurements: List[dict] = []
    unverified = _strs(raw.get("unverified_items"), 15)
    for m in _rows(raw.get("measurements")):
        item = _text(m.get("item"), 80)
        value = _num(m.get("value"))
        page = _int(m.get("page", m.get("source_page")))
        if not item:
            continue
        if value is None:
            unverified.append(f"{item} (p{page})" if page else item)
            continue
        measurements.append(
            {
                "item": item,
                "value": value,
                "unit": _text(m.get("unit"), 12),
                "page": page,
                "confidence": _conf(m.get("confidence")),
            }
        )
    return _carry_warning(raw, {"measurements": measurements[:15], "unverified_items": unverified[:15]})


# ============================================================
# AGENT 3: QS / BOQ
# ============================================================

def run_qs_agent(api_key, model, drawing_analysis, measurements, methodology, project_name, unit_system) -> Dict[str, Any]:
    drawing_brief = {
        "summary": drawing_analysis.get("summary", ""),
        "pages": [
            {k: p.get(k) for k in ("page", "type", "level", "scale")}
            for p in drawing_analysis.get("pages", [])
        ],
    }
    notes = "\n".join(
        f"- {m.get('text', '')[:450]}" for m in (methodology or []) if m.get("text")
    ) or "none"

    system = (
        "You are a senior quantity surveyor preparing a PRELIMINARY takeoff from measured evidence. "
        "Never invent dimensions. Quantities must come from the supplied measurements; if evidence is "
        "insufficient set quantity to null and say why in assumptions or excluded_items. "
        "Give a short formula using the supplied values. " + JSON_RULES
    )
    user = f"""Project: {project_name}. Units: {unit_system}.

DRAWINGS: {json.dumps(drawing_brief, ensure_ascii=False)}

MEASUREMENTS: {json.dumps(measurements.get("measurements", []), ensure_ascii=False)}

UNVERIFIED: {json.dumps(measurements.get("unverified_items", []), ensure_ascii=False)}

MEASUREMENT-RULE EXTRACTS (reference only, do not copy):
{notes}

Create a short BOQ (max 12 items) only for items the measurements support
(e.g. floor area, door/window counts, wall lengths). Item unit one of m, m2, m3, nr.
source_pages = pages of the measurements used.

Schema:
{{"items":[{{"item_no":1,"description":"","unit":"","quantity":null,"formula":"","source_pages":[],"confidence":"HIGH"}}],"assumptions":[],"excluded_items":[]}}"""

    raw = _run(
        _client(api_key),
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "qs",
    )

    items = []
    for i, it in enumerate(_rows(raw.get("items")), start=1):
        description = _text(it.get("description"), 100)
        if not description:
            continue
        items.append(
            {
                "item_no": len(items) + 1,
                "description": description,
                "unit": _text(it.get("unit"), 10),
                "quantity": _num(it.get("quantity")),
                "formula": _text(it.get("formula"), 120),
                "source_pages": _pages_list(it.get("source_pages")),
                "confidence": _conf(it.get("confidence")),
            }
        )
    result = {
        "items": items[:12],
        "assumptions": _strs(raw.get("assumptions"), 8),
        "excluded_items": _strs(raw.get("excluded_items"), 8),
    }
    return _carry_warning(raw, result)


# ============================================================
# AGENT 4: REVIEW (compact; never regenerates the BOQ)
# ============================================================

def run_review_agent(api_key, model, measurements, qs_result, auto_findings) -> Dict[str, Any]:
    boq_brief = [
        {k: it.get(k) for k in ("item_no", "description", "unit", "quantity", "formula", "source_pages", "confidence")}
        for it in qs_result.get("items", [])
    ]
    system = (
        "You are an independent QA reviewer of a preliminary quantity takeoff. Flag ONLY: unsupported "
        "quantities, missing evidence, calculation errors, low-confidence inputs, duplicate quantities. "
        "List only items with a problem. Do not rewrite the BOQ or add quantities. " + JSON_RULES
    )
    user = f"""MEASUREMENTS: {json.dumps(measurements.get("measurements", []), ensure_ascii=False)}

BOQ: {json.dumps(boq_brief, ensure_ascii=False)}

ALREADY FOUND BY AUTOMATIC CHECKS (do not repeat): {json.dumps(auto_findings.get("checks", []), ensure_ascii=False)}

Add only extra problems. status is REVIEW_REQUIRED or UNABLE_TO_VERIFY. issue max 12 words.

Schema:
{{"overall_status":"PASS","checks":[{{"item_no":1,"status":"REVIEW_REQUIRED","issue":""}}],"warnings":[]}}"""

    raw = _run(
        _client(api_key),
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "review",
    )

    checks = []
    for c in _rows(raw.get("checks")):
        status = str(c.get("status") or "REVIEW_REQUIRED").strip().upper()
        if status not in {"PASS", "REVIEW_REQUIRED", "UNABLE_TO_VERIFY"}:
            status = "REVIEW_REQUIRED"
        if status == "PASS":
            continue
        checks.append(
            {"item_no": _int(c.get("item_no")), "status": status, "issue": _text(c.get("issue"), 120)}
        )
    return _carry_warning(raw, {"checks": checks[:15], "warnings": _strs(raw.get("warnings", raw.get("critical_warnings")), 8)})
