# Product Catalog Research Agent 


It uses a **single unified LangGraph-backed agent** built with the modern `create_agent(...)` API.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate   
pip install -r requirements.txt
python -m app.main
```

## Tools included

1. `search_products(query)`
2. `get_product_details(product_id)`
3. `add_to_shortlist(product_id, quantity=1)`
4. `view_shortlist()`
5. `load_category_info(category)` *(async bonus tool)*

## Hybrid search design

`search_products` merges two strategies:

### 1) Semantic search
- Implemented with **TF-IDF + cosine similarity** over:
  - product name
  - product code
  - category
  - description

### 2) Keyword search
- Exact and partial string matching over:
  - product name
  - product code
  - description
- Includes token coverage, prefix bonus, and substring bonus.

### Combined scoring formula

```text
final_score =
    0.65 * semantic_score
  + 0.35 * keyword_score
  + exact_match_bonus
  + overlap_bonus
```

Where:
- `semantic_score`: cosine similarity from TF-IDF vectors
- `keyword_score`: lexical coverage-based score in [0, 1]
- `exact_match_bonus = 0.20` when query exactly matches product code or full product name
- `overlap_bonus = 0.10` when both semantic and keyword signals are present

## Example test prompts

- `find a laptop for students`
- `search novaphone`
- `show me LAP-ULTRA-14`
- `what are the audio products?`
- `add P005 quantity 2 to shortlist`
- `view shortlist`
