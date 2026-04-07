"""
hybrid_retriever.py — Hybrid retrieval coordinator
Integrates BM25 keyword retrieval and vector semantic retrieval into a unified interface,
executing dual-lane retrieval in parallel for each query variant as input to RRF fusion.

Retrieval workflow:
  For each query variant q_i:
    ├── BM25 retrieval → [bm25_docs_i]       (exact keyword matching)
    └── Vector retrieval → [vector_docs_i]   (semantic similarity matching)
  └──→ Fed into DualRRFFusion.fuse()

Vector retrieval supports metadata filtering (e.g., by section, tenant, importance),
complementing BM25 retrieval to ensure high recall.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.indexing.indexer import ChromaIndexer
from app.retrieval.rrf import DualRRFFusion, rrf_merge, deduplicate

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever: executes BM25 + vector dual-lane retrieval in parallel
    for multiple query variants, then fuses via DualRRFFusion, outputting the final candidate set.

    Retrieval architecture (two lanes):
      Lane 1: BM25 + Vector(children)  — main retrieval (keywords + semantics)
      Lane 2: Summary-guided Vector    — summary-guided section-level precision retrieval
                                         first finds relevant section summaries → then precise retrieval within section
      Both lanes are finally fused via RRF; Lane 2 silently degrades to no-op if summaries don't exist.

    Usage:
        retriever = HybridRetriever(indexer)
        results = await retriever.retrieve(
            query_variants={"original": q, "rewritten": q2, "hyde": q3, ...},
            tenant_id="default",
            filters=None,
        )
    """

    def __init__(self, indexer: ChromaIndexer):
        self.indexer = indexer
        self.fuser = DualRRFFusion()
        self._top_k = settings.retrieval_top_k  # Top-K per retrieval lane

    async def retrieve(
        self,
        query_variants: Dict[str, str],
        tenant_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval in parallel for all query variants, then fuses and deduplicates.

        Args:
            query_variants: {variant_name: query_text}, e.g.
                            {"original": "...", "rewritten": "...", "hyde": "..."}
            tenant_id: Tenant ID (data isolation)
            filters: Chroma metadata filter conditions (optional)
            top_n: Number of documents to return after fusion (optional, uses rrf_k result by default)

        Returns:
            Candidate document list sorted by RRF score descending
        """
        # ── Concurrently execute dual-lane retrieval for all variants ─────────
        # asyncio.gather runs all coroutines concurrently; total latency ≈ slowest single retrieval
        tasks = {
            variant_name: self._retrieve_single(    # Coroutine not yet running, just created
                query_text=query_text,
                variant_name=variant_name,
                tenant_id=tenant_id,
                filters=filters,
            )
            for variant_name, query_text in query_variants.items()
            if query_text  # Skip empty queries
        }

        # Concurrent execution (dict comprehension with asyncio.gather)
        variant_names = list(tasks.keys())
        coros = list(tasks.values())

        try:
            results_list = await asyncio.gather(*coros, return_exceptions=True)
        except Exception as e:
            logger.error(f"Concurrent retrieval error: {e}")
            return []

        # Organize into {variant_name: {"bm25": [...], "vector": [...]}} format
        query_results: Dict[str, Dict[str, List[Dict]]] = {}
        for name, result in zip(variant_names, results_list):
            if isinstance(result, Exception):
                logger.warning(f"Query variant [{name}] retrieval failed: {result}")
                query_results[name] = {"bm25": [], "vector": []}
            else:
                query_results[name] = result
                logger.debug(
                    f"[{name}] BM25: {len(result['bm25'])}, "
                    f"Vector: {len(result['vector'])}"
                )

        # ── Dual-layer RRF fusion ────────────────────────────────────────────
        final_docs = self.fuser.fuse(query_results, top_n=top_n)

        # ── Context expansion: replace child chunk results with parent chunk full text ──
        final_docs = await self._expand_to_parents(final_docs, tenant_id)
        # Deduplicate again by parent chunk_id (different child chunks may point to same parent)
        final_docs = deduplicate(final_docs)

        # ── Summary-guided retrieval (Lane 2) ────────────────────────────────
        # Use original query to find relevant sections in summary collection,
        # then do precise retrieval within section child chunks.
        # If summary collection doesn't exist (--with-summary not enabled), silently skip.
        original_query = next(iter(query_variants.values()), "")
        summary_guided = await self._retrieve_summary_guided(
            query_text=original_query,
            tenant_id=tenant_id,
            filters=filters,
            top_n=top_n or self._top_k,
        )
        if summary_guided:
            merged = rrf_merge([final_docs, summary_guided], k=settings.rrf_k)
            final_docs = deduplicate(merged)
            if top_n:
                final_docs = final_docs[:top_n]
            logger.info(f"Summary-guided added {len(summary_guided)} docs, total after fusion: {len(final_docs)}")

        logger.info(
            f"Hybrid retrieval complete: {len(query_variants)} query variants, "
            f"final {len(final_docs)} candidate documents"
        )
        return final_docs

    async def _retrieve_single(
        self,
        query_text: str,
        variant_name: str,
        tenant_id: str,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, List[Dict]]:
        """
        Executes BM25 + vector dual-lane retrieval for a single query.
        Uses asyncio.to_thread to convert synchronous CPU/IO-intensive operations to async,
        avoiding blocking FastAPI's event loop.

        Returns:
            {"bm25": [...], "vector": [...]}
        """
        # BM25 retrieval (sync, converted to async)
        bm25_task = asyncio.to_thread(
            self.indexer.bm25_search,
            query_text=query_text,
            tenant_id=tenant_id,
            top_k=self._top_k,
        )

        # Vector retrieval (sync, converted to async)
        vector_task = asyncio.to_thread(
            self.indexer.vector_search,
            query_text=query_text,
            tenant_id=tenant_id,
            top_k=self._top_k,
            where=self._build_chroma_filter(filters),
            collection_type="children",
        )

        # Execute both retrieval lanes concurrently
        bm25_docs, vector_docs = await asyncio.gather(bm25_task, vector_task)

        return {"bm25": bm25_docs, "vector": vector_docs}

    async def _retrieve_summary_guided(
        self,
        query_text: str,
        tenant_id: str,
        filters: Optional[Dict[str, Any]],
        top_n: int,
        top_k_summaries: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Summary-guided section-level retrieval (Lane 2):
          1. Vector retrieval in summary collection to find top_k_summaries most relevant sections
          2. Extract source_chunk_id (corresponding parent chunk ID) from summary metadata
          3. Do precise vector retrieval within child chunks of those parent chunks
          4. Expand child chunks to parent full text

        Returns [] without affecting main flow if summary collection doesn't exist
        (user didn't enable --with-summary when building the index).
        """
        # Step 1: Summary vector retrieval
        try:
            summary_docs = await asyncio.to_thread(
                self.indexer.vector_search,
                query_text=query_text,
                tenant_id=tenant_id,
                top_k=top_k_summaries,
                collection_type="summaries",
            )
        except Exception as e:
            logger.debug(f"Summary collection not found or retrieval error, skipping summary-guided: {e}")
            return []

        if not summary_docs:
            return []

        # Step 2: Extract parent chunk IDs from matching sections
        parent_ids = [
            doc["metadata"].get("source_chunk_id", "")
            for doc in summary_docs
            if doc["metadata"].get("source_chunk_id", "")
        ]
        if not parent_ids:
            return []

        logger.debug(
            f"Summary-guided matched {len(summary_docs)} sections: "
            f"{[d['metadata'].get('section_title', '') for d in summary_docs]}"
        )

        # Step 3: Precise vector retrieval within child chunks of matching sections
        # Chroma $in filter: only retrieve child chunks belonging to these parent chunks
        chroma_filter: Dict[str, Any] = {"parent_chunk_id": {"$in": parent_ids}}
        user_filter = self._build_chroma_filter(filters)
        if user_filter:
            chroma_filter = {"$and": [chroma_filter, user_filter]}

        try:
            guided_children = await asyncio.to_thread(
                self.indexer.vector_search,
                query_text=query_text,
                tenant_id=tenant_id,
                top_k=top_n,
                where=chroma_filter,
                collection_type="children",
            )
        except Exception as e:
            logger.warning(f"Summary-guided child chunk retrieval failed: {e}")
            return []

        if not guided_children:
            return []

        # Step 4: Expand to parent full text (reuse existing logic)
        return await self._expand_to_parents(guided_children, tenant_id)

    async def _expand_to_parents(
        self,
        child_docs: List[Dict[str, Any]],
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Parent-Child context expansion:
        Replaces/augments child chunk retrieval results with the full text of the corresponding parent chunk.
        Keeps child chunk RRF scores, uses parent chunk text to provide more complete context for LLM.

        If parent chunk not found (orphan child chunk), keeps the original child chunk.
        """
        expanded = []
        for doc in child_docs:
            meta = doc.get("metadata", {})
            parent_id = meta.get("parent_chunk_id", "")

            if parent_id and parent_id != "None" and parent_id != "":
                # Async fetch parent chunk full text
                parent = await asyncio.to_thread(
                    self.indexer.get_parent_by_id,
                    parent_id=parent_id,
                    tenant_id=tenant_id,
                )
                if parent:
                    # Replace child chunk text with parent text, keep child scores and metadata
                    expanded_doc = {
                        **doc,
                        "text": parent["text"],           # Parent full text (more complete)
                        "child_text": doc["text"],         # Keep original child text (for debugging)
                        "metadata": {**doc["metadata"], **parent["metadata"]},
                    }
                    expanded.append(expanded_doc)
                    continue

            # Parent not found, keep original child chunk
            expanded.append(doc)

        return expanded

    def _build_chroma_filter(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict]:
        """
        Converts API request filter conditions into Chroma where clause format.

        Chroma filter syntax example:
          {"chapter": "Cardiology"}
          → {"$and": [{"chapter": {"$eq": "Cardiology"}}]}

        Supported operators:
          {"field": value}                 → exact match
          {"field": {"$contains": "..."}}  → string contains
          {"$and": [...]}                  → AND
          {"$or": [...]}                   → OR
        """
        if not filters:
            return None

        # Simple filter: all conditions joined with AND
        conditions = []
        for field, value in filters.items():
            if isinstance(value, str):
                conditions.append({field: {"$eq": value}})
            elif isinstance(value, dict):
                # Pass through complex operators (e.g., {"$contains": "..."})
                conditions.append({field: value})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]  # No need to wrap single condition in $and
        return {"$and": conditions}
