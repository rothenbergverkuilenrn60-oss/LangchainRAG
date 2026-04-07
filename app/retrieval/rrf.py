"""
rrf.py — Reciprocal Rank Fusion (RRF)
Implements the dual-layer RRF architecture used in this system:

  Layer 1 RRF (per-query level):
    For each query variant, fuses BM25 results and vector retrieval results
    → produces a "hybrid ranking list" for that query

  Layer 2 RRF (cross-query level):
    Fuses the hybrid ranking lists from all query variants again
    → produces the final candidate set

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
  - k: smoothing constant (default 60, reduces absolute advantage of top-ranked docs)
  - rank_i(d): rank of document d in the i-th ranking list (starting from 1)
  - If document is not in a list, its contribution is 0

Design advantages:
  - Independent of score magnitude (BM25 and cosine similarity have different scales)
  - Robust to noise (false positives from a single retrieval system have limited impact on fusion)
  - O(n) time complexity, extremely fast
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Document unique key extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_doc_key(doc: Dict[str, Any]) -> str:
    """
    Extracts a unique identifier from a retrieval result dict.
    Prefers chunk_id (exact match), falls back to text hash (avoids duplicates).
    """
    meta = doc.get("metadata", {})
    chunk_id = meta.get("chunk_id", "")
    if chunk_id:
        return chunk_id
    # Fallback: MD5 hash of first 100 characters of text
    import hashlib
    text_snippet = doc.get("text", "")[:100]
    return hashlib.md5(text_snippet.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# RRF core function
# ─────────────────────────────────────────────────────────────────────────────

def rrf_merge(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = None,
    top_n: int = None,
) -> List[Dict[str, Any]]:
    """
    Fuses multiple sorted retrieval result lists into a single unified ranking.

    Args:
        ranked_lists: Multiple sorted lists, each sorted by relevance descending.
                      Each document is a dict with at least "text" and "metadata".
        k: RRF smoothing parameter (reads from config by default)
        top_n: Return top N results (None = return all)

    Returns:
        Document list sorted by RRF score descending, each document gains a "rrf_score" field
    """
    k = k or settings.rrf_k

    # {doc_key: {"doc": ..., "rrf_score": float, "appearances": int, "ranks": []}}
    doc_scores: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "doc": None,
        "rrf_score": 0.0,
        "appearances": 0,
        "ranks": [],
    })

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked_list):
            doc_key = _get_doc_key(doc)
            # RRF formula: 1 / (k + rank), rank starts from 1
            rrf_contribution = 1.0 / (k + rank + 1)

            entry = doc_scores[doc_key]     # Get or create entry for this document
            if entry["doc"] is None:
                # First encounter of this document, record its original data
                entry["doc"] = doc
            entry["rrf_score"] += rrf_contribution
            entry["appearances"] += 1
            entry["ranks"].append((list_idx, rank + 1))

    # Sort by RRF score descending
    sorted_items = sorted(
        doc_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # Build final result list
    merged: List[Dict[str, Any]] = []
    for item in sorted_items:
        doc = dict(item["doc"])  # Shallow copy to avoid modifying original data
        doc["rrf_score"] = round(item["rrf_score"], 6)
        doc["rrf_appearances"] = item["appearances"]    # Number of lists this doc appeared in
        doc["rrf_ranks"] = item["ranks"]                # Rank info per list
        # Use RRF score as the primary score
        doc["score"] = doc["rrf_score"]
        merged.append(doc)

    if top_n:
        merged = merged[:top_n]

    logger.debug(
        f"RRF fusion: {len(ranked_lists)} lists, "
        f"total docs {sum(len(l) for l in ranked_lists)}, "
        f"after dedup {len(doc_scores)}, output {len(merged)}"
    )
    return merged


def deduplicate(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates based on chunk_id (keeps the one with the highest RRF score).
    Called between two RRF passes and before Reranker to ensure no duplicate documents.

    Args:
        docs: Document list (already sorted by score descending)

    Returns:
        Deduplicated document list (preserves original order)
    """
    seen_keys = set()
    unique_docs = []
    for doc in docs:
        key = _get_doc_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(doc)
    logger.debug(f"Deduplication: {len(docs)} → {len(unique_docs)} documents")
    return unique_docs


# ─────────────────────────────────────────────────────────────────────────────
# Dual-layer RRF wrapper class
# ─────────────────────────────────────────────────────────────────────────────

class DualRRFFusion:
    """
    Dual-layer RRF fuser, encapsulating the complete fusion workflow:

    Layer 1: per-query hybrid (BM25 + vector → hybrid ranking)
    Layer 2: cross-query aggregation (all variant results → final candidate set)
    + Deduplication (by chunk_id, prevents the same fragment from appearing multiple times)

    Usage:
        fuser = DualRRFFusion()
        final_docs = fuser.fuse(query_results_map)
        # query_results_map = {
        #   "original": {"bm25": [...], "vector": [...]},
        #   "rewritten": {"bm25": [...], "vector": [...]},
        #   ...
        # }
    """

    def __init__(self, k: int = None):
        self.k = k or settings.rrf_k

    def fuse(
        self,
        query_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes dual-layer RRF fusion.

        Args:
            query_results: {query_variant_name: {"bm25": [...], "vector": [...]}}
            top_n: Final number of documents to return

        Returns:
            Document list sorted by final RRF score descending
        """
        # ── Layer 1: Within each query variant, BM25 + vector → hybrid ranking ──
        layer1_results: List[List[Dict[str, Any]]] = []

        for variant_name, retrieval_dict in query_results.items():
            bm25_docs = retrieval_dict.get("bm25", [])
            vector_docs = retrieval_dict.get("vector", [])

            if not bm25_docs and not vector_docs:
                continue

            # Layer 1 RRF: fuse BM25 and vector results for this query
            hybrid = rrf_merge(
                ranked_lists=[bm25_docs, vector_docs],
                k=self.k,
            )
            # Add source marker to fusion results for debugging
            for doc in hybrid:
                doc["source_query"] = variant_name

            layer1_results.append(hybrid)
            logger.debug(
                f"Layer1 RRF [{variant_name}]: "
                f"BM25({len(bm25_docs)}) + Vector({len(vector_docs)}) "
                f"→ {len(hybrid)} docs"
            )

        if not layer1_results:
            logger.warning("All query variants returned no retrieval results")
            return []

        # ── Layer 2: Cross-query variant aggregation ─────────────────────────
        layer2_merged = rrf_merge(
            ranked_lists=layer1_results,
            k=self.k,
        )
        logger.info(
            f"Layer2 RRF: {len(layer1_results)} lanes → {len(layer2_merged)} candidate docs"
        )

        # ── Deduplication (at chunk_id level) ────────────────────────────────
        deduplicated = deduplicate(layer2_merged)
        logger.info(f"After deduplication: {len(deduplicated)} unique documents")

        # Truncate to top_n
        if top_n:
            deduplicated = deduplicated[:top_n]

        return deduplicated
