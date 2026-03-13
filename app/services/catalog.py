from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"
print(DATA_PATH)

class CatalogService:
    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = data_path or DATA_PATH
        self._products = self._load_products()

    def _load_products(self) -> List[Dict[str, Any]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def products(self) -> List[Dict[str, Any]]:
        return self._products

    def get_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        for product in self._products:
            if product["id"].lower() == product_id.lower():
                return product
        return None

    def get_by_code_or_name(self, value: str) -> Optional[Dict[str, Any]]:
        needle = value.strip().lower()
        for product in self._products:
            if product["code"].lower() == needle or product["name"].lower() == needle:
                return product
        return None

    def get_catalog_summary(self) -> Dict[str, Any]:
        categories: Dict[str, int] = {}
        for product in self._products:
            categories[product["category"]] = categories.get(product["category"], 0) + 1
        return {
            "total_products": len(self._products),
            "categories": categories,
        }

    def get_category_info(self, category: str) -> Dict[str, Any]:
        items = [p for p in self._products if p["category"].lower() == category.lower()]
        print(self._products)
        print(items)
        if not items:
            return {"status": "not_found", "message": f"No category named '{category}' found."}
        return {
            "status": "ok",
            "category": category,
            "count": len(items),
            "products": items,
        }
