from __future__ import annotations

#import math
import re
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import CatalogService



class HybridSearchService:
    """Hybrid search using TF-IDF semantic similarity + keyword matching.

    Scoring formula:
        final_score =
            0.65 * semantic_score
            + 0.35 * keyword_score
            + exact_match_bonus
            + overlap_bonus

    Where:
      - semantic_score: cosine similarity on name + category + description + code
      - keyword_score: weighted lexical score from exact token matches/substrings
      - exact_match_bonus: +0.20 when query exactly matches product code, name, or category
      - overlap_bonus: +0.10 when item appears in both semantic and keyword results

    Thresholds:
      - >= 0.88 and score gap >= 0.08 from second result => auto-select
      - otherwise if results exist => needs clarification
      - 0 results => not_found
    """

    def __init__(self, catalog_service: CatalogService) -> None:
        self.catalog = catalog_service
        self.products = catalog_service.products

        self.documents = [
            " ".join([
                p["name"],
                p["code"],
                p["category"],
                p["description"],
            ])
            for p in self.products
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.product_matrix = self.vectorizer.fit_transform(self.documents)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _normalize(self, text: str) -> str:
        return " ".join(self._tokenize(text))

    def _keyword_score(self, query: str, product: Dict[str, Any]) -> float:
        print(f"query: {query}")
        q = self._normalize(query)
        tokens = self._tokenize(query)

        if not tokens:
            return 0.0

        name = self._normalize(product["name"])
        code = self._normalize(product["code"])
        desc = self._normalize(product["description"])
        category = self._normalize(product["category"])

        #Higher importance fields
        name_hits = sum(1 for t in tokens if t in name)
        code_hits = sum(1 for t in tokens if t in code)

        #Lower importance fields
        desc_hits = sum(1 for t in tokens if t in desc)
        category_hits = sum(1 for t in tokens if t in category)

        #Weighted token hit score
        weighted_hits = (
            1.00 * name_hits +
            1.20 * code_hits +
            0.45 * desc_hits +
            0.60 * category_hits
        )

        max_possible = max(len(tokens) * 1.20, 1.0)
        coverage = min(1.0, weighted_hits / max_possible)

        exact = 1.0 if q in {name, code, category} else 0.0

        prefix_bonus = 0.15 if (
            name.startswith(q) or
            code.startswith(q) or
            category.startswith(q)
        ) else 0.0

        substring_bonus = 0.10 if q and (
            q in name or
            q in code or
            q in category
        ) else 0.0

        #Extra bonus if query token set strongly matches category intent
        #Example: "laptop", "skincare", "monitor"
        category_token_overlap = category_hits / max(len(tokens), 1)
        category_intent_bonus = 0.08 if category_token_overlap >= 0.6 else 0.0

        score = (
            0.55 * coverage +
            0.20 * exact +
            prefix_bonus +
            substring_bonus +
            category_intent_bonus
        )

        return min(1.0, score)

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "status": "not_found",
                "message": "Please provide a product name, code, or descriptive phrase.",
                "results": [],
            }

        query_text = query.strip()
        normalized_query = self._normalize(query_text)

        qvec = self.vectorizer.transform([query_text])
        semantic_scores = cosine_similarity(qvec, self.product_matrix)[0]

        ranked: List[Dict[str, Any]] = []

        for idx, product in enumerate(self.products):
            print(f"products: {product}")
            semantic_score = float(max(0.0, semantic_scores[idx]))
            print(f"semantic score :{semantic_score}")
            keyword_score = self._keyword_score(query_text, product)
            print(f"keyword score :{keyword_score}")
            product_name = self._normalize(product["name"])
            product_code = self._normalize(product["code"])
            product_category = self._normalize(product["category"])

            exact_match_bonus = 0.20 if normalized_query in {
                product_name,
                product_code,
                product_category,
            } else 0.0

            overlap_bonus = 0.10 if semantic_score > 0.05 and keyword_score > 0.05 else 0.0

            final_score = (
                0.65 * semantic_score +
                0.35 * keyword_score +
                exact_match_bonus +
                overlap_bonus
            )
            final_score = min(1.0, final_score)

            if final_score >= 0.30:
                ranked.append({
                    "product": product,
                    "semantic_score": round(semantic_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "final_score": round(final_score, 4),
                })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)

        if not ranked:
            return {
                "status": "not_found",
                "message": f"I couldn't find a product matching '{query_text}'. Try a product code, a shorter name, or a category.",
                "results": [],
            }

        top = ranked[0]
        second_score = ranked[1]["final_score"] if len(ranked) > 1 else 0.0
        score_gap = top["final_score"] - second_score

        if top["final_score"] >= 0.88 and score_gap >= 0.08:
            return {
                "status": "auto_selected",
                "message": f"Auto-selected {top['product']['name']} based on a high-confidence match.",
                "selected_product": top["product"],
                "results": ranked[:top_k],
                "confidence": top["final_score"],
            }

        return {
            "status": "needs_clarification",
            "message": "I found multiple possible matches. Please choose one of these options.",
            "choices": [
                {
                    "id": item["product"]["id"],
                    "code": item["product"]["code"],
                    "name": item["product"]["name"],
                    "category": item["product"]["category"],
                    "score": item["final_score"],
                }
                for item in ranked[:top_k]
            ],
            "results": ranked[:top_k],
            "confidence": top["final_score"],
        }