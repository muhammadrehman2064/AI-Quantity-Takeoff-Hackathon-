from typing import Any, Dict, List


def build_boq(qs_result: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    """Merge QS output with reviewer status without changing quantities."""
    review_by_item = {}

    for check in review.get("checks", []):
        item_no = check.get("item_no")
        if item_no is not None:
            review_by_item[str(item_no)] = check

    items = []
    for item in qs_result.get("items", []):
        item_copy = dict(item)
        key = str(item_copy.get("item_no"))
        check = review_by_item.get(key)

        if check:
            item_copy["review_status"] = check.get("status")
            item_copy["review_issue"] = check.get("issue", "")
        else:
            item_copy["review_status"] = "NOT_REVIEWED"

        items.append(item_copy)

    return {
        "status": "PRELIMINARY",
        "items": items,
        "assumptions": qs_result.get("assumptions", []),
        "excluded_items": qs_result.get("excluded_items", []),
        "review_status": review.get("overall_status", "REVIEW_REQUIRED"),
    }
