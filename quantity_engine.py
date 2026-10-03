import ast
import re
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# Deterministic checks (no AI tokens)
# ------------------------------------------------------------

def _eval_formula(formula: str) -> Optional[float]:
    """Evaluate a purely numeric formula like '4.2 x 3.6' or '(5.2*2)+3 = 13.4'. None if not numeric."""
    expr = (formula or "").split("=")[0]
    expr = re.sub(r"(?<=[\d)])\s*[xX×]\s*(?=[\d(])", "*", expr)
    expr = re.sub(r"[A-Za-z²³]+", "", expr).strip()
    if not expr or not re.fullmatch(r"[\d\s.+\-*/()]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)
        if not all(isinstance(n, allowed) for n in ast.walk(tree)):
            return None
        return float(eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def precheck_boq(measurements: Dict[str, Any], qs_result: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap, reliable review checks: missing evidence, arithmetic, low confidence, duplicates."""
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen: Dict[tuple, int] = {}

    for it in qs_result.get("items", []):
        no, qty = it.get("item_no"), it.get("quantity")
        issues: List[str] = []

        if qty is not None:
            if qty <= 0:
                issues.append("Non-positive quantity")
            if not (it.get("formula") or "").strip():
                issues.append("No formula supporting quantity")
            if not it.get("source_pages"):
                issues.append("No source page")
            calc = _eval_formula(it.get("formula", ""))
            if calc is not None and abs(calc - qty) > max(0.02 * abs(qty), 0.01):
                issues.append(f"Formula gives {calc:.2f}, quantity is {qty:g}")
        if it.get("confidence") == "LOW":
            issues.append("Low-confidence evidence")

        key = ((it.get("description") or "").strip().lower(), (it.get("unit") or "").strip().lower())
        if key[0] and key in seen:
            issues.append(f"Possible duplicate of item {seen[key]}")
        else:
            seen[key] = no

        if issues:
            checks.append({"item_no": no, "status": "REVIEW_REQUIRED", "issue": "; ".join(issues)})

    rows = measurements.get("measurements", [])
    low = sum(1 for m in rows if m.get("confidence") == "LOW")
    if low:
        warnings.append(f"{low} low-confidence measurement(s) used")
    if not rows:
        warnings.append("No reliable measurements were extracted")
    if not qs_result.get("items"):
        warnings.append("No BOQ items could be supported by the evidence")
    elif all(it.get("quantity") is None for it in qs_result["items"]):
        warnings.append("No BOQ item has a supported quantity")

    return {"checks": checks, "warnings": warnings}


def merge_reviews(auto: Dict[str, Any], ai: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine automatic checks with the AI reviewer's extras into one review dict."""
    merged: Dict[Any, Dict[str, Any]] = {}
    general_issues: List[Dict[str, Any]] = []

    for check in auto.get("checks", []) + ((ai or {}).get("checks", [])):
        no = check.get("item_no")
        if no is None:
            general_issues.append(check)
            continue
        if no in merged:
            prev = merged[no]
            if check.get("issue") and check["issue"] not in prev["issue"]:
                prev["issue"] = f"{prev['issue']}; {check['issue']}".strip("; ")
            if check.get("status") == "REVIEW_REQUIRED":
                prev["status"] = "REVIEW_REQUIRED"
        else:
            merged[no] = dict(check)

    warnings = list(auto.get("warnings", []))
    for w in (ai or {}).get("warnings", []):
        if w not in warnings:
            warnings.append(w)
    for g in general_issues:
        if g.get("issue"):
            warnings.append(g["issue"])

    checks = [merged[k] for k in sorted(merged, key=lambda x: (str(type(x)), x))]
    review = {
        "overall_status": "REVIEW_REQUIRED" if (checks or warnings) else "PASS",
        "checks": checks,
        "warnings": warnings,
        "ai_review": ai is not None,
    }
    if ai and ai.get("_warning"):
        review["note"] = ai["_warning"]
    return review


# ------------------------------------------------------------
# BOQ assembly
# ------------------------------------------------------------

def build_boq(qs_result: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    """Merge QS items with review status without changing quantities."""
    by_item = {str(c.get("item_no")): c for c in review.get("checks", []) if c.get("item_no") is not None}
    default_status = "PASS" if review.get("ai_review") else "AUTO_CHECKED"

    items = []
    for item in qs_result.get("items", []):
        row = dict(item)
        check = by_item.get(str(row.get("item_no")))
        row["review_status"] = check.get("status") if check else default_status
        row["review_issue"] = check.get("issue", "") if check else ""
        items.append(row)

    return {
        "status": "PRELIMINARY",
        "items": items,
        "assumptions": qs_result.get("assumptions", []),
        "excluded_items": qs_result.get("excluded_items", []),
        "review_status": review.get("overall_status", "REVIEW_REQUIRED"),
    }
