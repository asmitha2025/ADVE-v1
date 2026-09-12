"""
bench — the experiments that decide whether this project has a product.

Run them in order. Each one can kill the next.

  Gate 1  bench/parity.py         Does skipping frames preserve retrieval?
  Gate 2  bench/routing_bench.py  Does change-aware routing beat uniform
                                  sampling at the SAME call budget?
  Gate 3  bench/routing_bench.py  Is the pipeline actually cheaper, measured
                                  in seconds and calls rather than FLOPs?

Gate 1 first, always. Every accuracy claim in this repo's docs/ folder is
cosine similarity to a ground-truth CLIP vector, which is a metric that
cannot fail: unrelated frames from one fixed camera sit at 0.85-0.95 in CLIP
space because the background dominates. Retrieval parity is the metric that
can fail, and it has never been measured here.

Publish whatever these produce. A published negative result makes you the
person who measured it; an unpublished positive one convinces nobody.
"""

from .queries import (
    QuerySet, DEFAULT_QUERIES, DOMAIN_QUERIES, queries_for,
    new_query_set, scaffold_repo_query_sets,
)

__all__ = [
    "QuerySet", "DEFAULT_QUERIES", "DOMAIN_QUERIES", "queries_for",
    "new_query_set", "scaffold_repo_query_sets",
]
