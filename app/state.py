from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import Annotated,TypedDict

from langgraph.graph.message import add_messages


def add_shortlist_items(
    current: Optional[List[Dict[str, Any]]],
    incoming: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Custom reducer for shortlist items.

    Rules:
    - Append new items
    - Merge duplicate IDs by summing quantities
    - If any incoming item contains replace=True, reset before applying
    """
    current = list(current or [])
    incoming = list(incoming or [])

    if not incoming:
        return current

    should_replace = any(bool(item.get("replace")) for item in incoming)
    if should_replace:
        current = []

    merged: Dict[str, Dict[str, Any]] = {item["id"]: dict(item) for item in current if item.get("id")}

    for item in incoming:
        if item.get("replace"):
            continue

        item_id = item.get("id")
        if not item_id:
            continue

        quantity = int(item.get("quantity", 1))
        if item_id in merged:
            merged[item_id]["quantity"] = int(merged[item_id].get("quantity", 0)) + quantity
        else:
            clean_item = dict(item)
            clean_item["quantity"] = quantity
            merged[item_id] = clean_item

    return list(merged.values())


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    selected_product: Optional[Dict[str, Any]]
    shortlist: Annotated[Optional[List[Dict[str, Any]]], add_shortlist_items]
    tool_log: List[Dict[str, Any]]
    total_estimated_tokens: int
