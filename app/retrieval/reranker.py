"""
reranker.py — BGE-Reranker-v2-M3 cross-encoder reranking
The reranker is a key quality improvement point in RAG.

Bi-encoder model (e.g., BGE-M3):
  - Query and document are encoded separately into vectors
  - Relevance computed via cosine similarity
  - Very fast, but precision limited by independent encoding

Cross-encoder (e.g., BGE-Reranker):
  - Query and document are concatenated and fed into Transformer together
  - Model directly outputs a relevance score (more accurate)
  - Slow (cannot pre-compute), used only for fine-ranking Top-K candidates

This module:
  1. Loads BGE-Reranker-v2-M3 (multilingual, supports Chinese and English)
  2. Fine-ranks the candidate set output from dual-layer RRF
  3. Returns Top-N high-quality results with precise relevance scores
"""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from app.config import settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    BGE-Reranker-v2-M3 reranker.

    Uses the sentence-transformers CrossEncoder interface,
    supporting precise relevance scoring for Chinese/English mixed text.

    Usage:
        reranker = BGEReranker()
        ranked_chunks = reranker.rerank(
            query="What drugs are used for hypertension?",
            docs=[...],       # Candidate document list from RRF output
            top_n=8,
        )
    """

    def __init__(self):
        self._model: Optional[CrossEncoder] = None
        self._model_path = settings.reranker_model_path

    def _load_model(self):
        """Lazily loads the CrossEncoder model (avoids VRAM usage at startup)"""
        if self._model is not None:
            return

        logger.info(f"Loading reranker model: {self._model_path}")
        try:
            self._model = CrossEncoder(
                model_name_or_path=self._model_path,
                # Use GPU (CUDA), automatically falls back to CPU if unavailable
                device="cuda",
                # max_length controls input length (truncated if exceeded)
                max_length=512,
            )
            logger.info("Reranker model loaded successfully (GPU)")
        except Exception as e:
            logger.warning(f"GPU loading failed, trying CPU: {e}")
            self._model = CrossEncoder(
                model_name_or_path=self._model_path,
                device="cpu",
                max_length=512,
            )
            logger.info("Reranker model loaded successfully (CPU)")

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[RetrievedChunk]:
        """
        Reranks candidate documents.

        Args:
            query: Original user question (paired with each document for scoring)
            docs: Candidate document list from RRF output, each dict has "text" and "metadata"
            top_n: Number of results to return (reads from config by default)

        Returns:
            List[RetrievedChunk], sorted by rerank_score descending
        """
        if not docs:
            return []

        top_n = top_n or settings.rerank_top_n
        self._load_model()

        # ── Build (query, document) input pairs ──────────────────────────────
        # CrossEncoder.predict() accepts [(query, doc), ...] format
        pairs = [(query, doc["text"]) for doc in docs]

        # ── Batch inference ──────────────────────────────────────────────────
        try:
            # predict() returns relevance score for each pair (logit value, higher = more relevant)
            scores = self._model.predict(
                pairs,
                batch_size=64,          # Batch inference, balances speed and VRAM
                show_progress_bar=False,
                convert_to_numpy=True,  # Return numpy array for easy sorting
            )
        except Exception as e:
            logger.error(f"Reranker inference failed: {e}")
            # Fallback: keep original RRF score ordering
            return self._fallback_to_rrf(docs, top_n)

        # ── Sort by rerank score ─────────────────────────────────────────────
        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True,  # Descending: most relevant first
        )

        # ── Build RetrievedChunk list ────────────────────────────────────────
        results: List[RetrievedChunk] = []
        for i, (score, doc) in enumerate(scored_docs[:top_n]):
            meta = doc.get("metadata", {})

            # Parse JSON string fields from metadata
            page_numbers = self._parse_json_field(meta.get("page_numbers", "[]"), [])
            entities = self._parse_json_field(meta.get("entities", "{}"), {})

            results.append(RetrievedChunk(
                chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                text=doc.get("text", ""),
                score=float(score),             # Final score uses rerank score
                rrf_score=doc.get("rrf_score", 0.0),
                rerank_score=float(score),      # Precise rerank score
                bm25_rank=doc.get("bm25_rank", 0),
                vector_rank=doc.get("vector_rank", 0),
                page_numbers=page_numbers,
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                doc_source=meta.get("doc_source", ""),
                entities=entities,
                metadata={
                    "source_query": doc.get("source_query", ""),
                    "rrf_appearances": doc.get("rrf_appearances", 1),
                    "importance_boost": meta.get("importance_boost", 1.0),
                    "has_warning": meta.get("has_warning", False),
                    "has_table": meta.get("has_table", False),
                    # Pass through figure chunk fields
                    "is_figure": meta.get("is_figure", False),
                    "image_path": meta.get("image_path", ""),
                    "figure_caption": meta.get("figure_caption", ""),
                    "figure_number": meta.get("figure_number", ""),
                },
            ))

        logger.info(
            f"Reranker complete: {len(docs)} candidates → Top {len(results)} "
            f"(highest score: {float(scored_docs[0][0]):.3f}, "
            f"lowest score: {float(scored_docs[min(top_n,len(scored_docs))-1][0]):.3f})"
        )
        return results

    def _fallback_to_rrf(
        self,
        docs: List[Dict[str, Any]],
        top_n: int,
    ) -> List[RetrievedChunk]:
        """Fallback: use RRF scores for ordering when reranker fails"""
        logger.warning("Using RRF scores as fallback ordering")
        results = []
        for i, doc in enumerate(docs[:top_n]):
            meta = doc.get("metadata", {})
            results.append(RetrievedChunk(
                chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                text=doc.get("text", ""),
                score=doc.get("rrf_score", 0.0),
                rrf_score=doc.get("rrf_score", 0.0),
                rerank_score=0.0,
                page_numbers=self._parse_json_field(meta.get("page_numbers", "[]"), []),
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                doc_source=meta.get("doc_source", ""),
                metadata={
                    "is_figure": meta.get("is_figure", False),
                    "image_path": meta.get("image_path", ""),
                    "figure_caption": meta.get("figure_caption", ""),
                    "figure_number": meta.get("figure_number", ""),
                },
            ))
        return results

    @staticmethod
    def _parse_json_field(value: Any, default: Any) -> Any:
        """Safely parses JSON string fields from Chroma metadata"""
        import json
        if isinstance(value, (list, dict)):
            return value  # Already the correct type
        if isinstance(value, str) and value:
            try:
                return json.loads(value)
            except Exception:
                return default
        return default
