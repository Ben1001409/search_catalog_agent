from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

from services.catalog import CatalogService
from services.search import HybridSearchService
from state import AgentState

catalog_service = CatalogService()
search_service = HybridSearchService(catalog_service)


@tool
def search_products(query: str, runtime: ToolRuntime[None, AgentState]) -> Command:
    """Search the catalog using hybrid semantic + keyword search.

    Use this whenever the user asks to find, search, browse, or identify a product.
    If the result is high confidence, this tool auto-selects the product in state.
    If ambiguous, it returns structured choices and the agent must present them as-is.
    """
    result = search_service.search(query)
    print(result)
    if result["status"] == "auto_selected":
        selected = result["selected_product"]
        return Command(update={
            "selected_product": selected,
            "messages": [
                ToolMessage(
                    content=result["message"],
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        })

    return Command(update={
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id=runtime.tool_call_id,
            )
        ]
    })


@tool
def get_product_details(product_id: str) -> Dict[str, Any]:
    """Get full details for a product by product ID.

    Use after the user chooses a product from clarification options.
    """
    product = catalog_service.get_by_id(product_id)
    if not product:
        return {
            "status": "error",
            "message": f"No product found with ID '{product_id}'. Ask the user to choose one of the listed IDs.",
        }
    return {
        "status": "ok",
        "product": product,
    }


@tool
def add_to_shortlist(product_code: str, runtime: ToolRuntime) -> Command:
    """Add a product to the shortlist by product code."""
    
    state = runtime.state
    shortlist = state.get("shortlist", [])
    #catalog_service = runtime.context["catalog_service"]
    #products = catalog_service.products
    products = catalog_service.products
    #products = state.get("products", [])
    print(f"products: {products}")
    print(f"product code: {product_code}")
    product = next((p for p in products if p["code"] == product_code), None)
    print(f"product: {product}")
    if not product:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Product with code '{product_code}' was not found.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if any(p["code"] == product_code for p in shortlist):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"{product['name']} is already in the shortlist.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    updated_shortlist = shortlist + [product]

    return Command(
        update={
            "shortlist": updated_shortlist,
            "messages": [
                ToolMessage(
                    content=f"Added {product['name']} ({product['code']}) to shortlist.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )

@tool
def view_shortlist(runtime: ToolRuntime[None, AgentState]) -> Dict[str, Any]:
    """View the current shortlist stored in state."""
    shortlist = runtime.state.get("shortlist") or []
    return {
        "status": "ok",
        "items": shortlist,
        "count": len(shortlist),
    }


@tool
async def load_category_info(category: str) -> Dict[str, Any]:
    """Load full category information on demand.

    This is the async tool used for progressive disclosure.
    """
    await asyncio.sleep(0.05)
    return catalog_service.get_category_info(category)
