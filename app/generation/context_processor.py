"""
context_processor.py — Context processor
Performs two key processing steps before feeding retrieval results to the LLM:

1. Context Compression:
   Filters out fragments with low relevance to the question, reducing LLM input tokens
   and mitigating "Lost in the Middle" effects and hallucination risk.

2. Long Context Selection:
   Research shows LLM attention is lowest for content in the middle of a prompt.
   The most relevant documents should be placed at the beginning or end.
   → Reorders reranked documents in a "high-low-high" alternating pattern.

3. Citation Index building:
   Assigns a citation number to each retained chunk for use in [Source X] annotations.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from app.models import RetrievedChunk, CitationSource
from app.config import settings

logger = logging.getLogger(__name__)


class ContextProcessor:
    """
    Context processor: compresses and reorders the document list output from Reranker,
    producing a prompt context string and citation info suitable for the LLM.

    Usage:
        processor = ContextProcessor()
        context_str, citations = processor.process(reranked_chunks, query)
    """

    # Minimum relevance threshold (chunks with rerank score below this are filtered out)
    # BGE-Reranker-v2-M3 outputs logit values; positive = relevant, negative = irrelevant
    RELEVANCE_THRESHOLD = -5.0

    # Maximum total context characters (DeepSeek supports 128K context; relaxed to fit more results)
    MAX_CONTEXT_CHARS = 20000

    def process(
        self,
        chunks: List[RetrievedChunk],
        query: str,
    ) -> Tuple[str, List[CitationSource]]:
        """
        Main workflow: compress → reorder → format.

        Args:
            chunks: RetrievedChunk list from Reranker (sorted by rerank_score descending)
            query: Original user question (used for relevance judgment)

        Returns:
            (context_str, citations)
            - context_str: Formatted context string passed to the LLM
            - citations: Fine-grained citation list
        """
        # Step 1: Relevance filtering (compression)
        filtered = self._compress(chunks)

        # Step 2: Long context reordering (mitigates "Lost in the Middle")
        reordered = self._reorder_for_llm(filtered)

        # Step 3: Character count limit (hard truncation to prevent context overflow)
        truncated = self._truncate(reordered)

        # Step 4: Build citation info
        citations = self._build_citations(truncated)

        # Step 5: Format into LLM prompt context string
        context_str = self._format_context(truncated)

        logger.info(
            f"Context processing: {len(chunks)} → filtered {len(filtered)} "
            f"→ truncated {len(truncated)} chunks, total chars {len(context_str)}"
        )
        return context_str, citations

    # ─────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────

    def _compress(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Context compression: filters out chunks with very low relevance.
        Uses rerank_score threshold, keeping only reliable quality chunks.
        """
        filtered = [
            c for c in chunks
            if c.rerank_score >= self.RELEVANCE_THRESHOLD
        ]
        # Ensure at least 1 chunk is kept (prevent empty context from filtering all chunks)
        if not filtered and chunks:
            filtered = [chunks[0]]
            logger.warning("All chunks below relevance threshold; force-keeping the highest-scored chunk")

        logger.debug(f"Compression filter: {len(chunks)} → {len(filtered)} chunks")
        return filtered

    def _reorder_for_llm(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        "Lost in the Middle" mitigation strategy:
        Places the most relevant documents at the beginning and end,
        and less relevant ones in the middle.

        For example, for documents ranked [1,2,3,4,5,6] (1 = most relevant):
        Reordered as [1, 3, 5, 6, 4, 2] — strongest at head and tail, weaker in middle.

        Rationale: LLM attention is strongest at the head and tail of the prompt.
        """
        if len(chunks) <= 2:
            return chunks  # Too few chunks, no need to reorder

        # Split chunks by even/odd index position
        # Odd indices (0, 2, 4...) → front half (most relevant first)
        # Even indices (1, 3, 5...) → back half (reversed, less relevant last)
        front_group = chunks[::2]   # Indices 0, 2, 4, 6, ...
        back_group = chunks[1::2]   # Indices 1, 3, 5, 7, ...

        # Reverse back half so less relevant chunks are near the end
        reordered = front_group + list(reversed(back_group))

        logger.debug(f"Long context reorder: {[round(c.rerank_score,2) for c in chunks[:3]]}... → reorder complete")
        return reordered

    def _truncate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Hard truncation by total character count to stay within LLM context window"""
        total_chars = 0
        result = []
        for chunk in chunks:
            chunk_len = len(chunk.text)
            if total_chars + chunk_len > self.MAX_CONTEXT_CHARS:
                # Limit exceeded, stop adding
                logger.debug(
                    f"Character limit truncation: keeping {len(result)} chunks ({total_chars} chars)"
                )
                break
            result.append(chunk)
            total_chars += chunk_len
        return result

    # Matches figure references like "图4-2", "图 4-2", "图4—2", etc.
    _FIGURE_REF_RE = re.compile(r'图\s*\d+[-—–]\d+')

    def _image_url(self, image_path: str) -> Optional[str]:
        """Converts an absolute image path to an HTTP URL accessible by the frontend."""
        if not image_path:
            return None
        try:
            rel = Path(image_path).relative_to(Path(settings.pic_dir))
            return f"/static/pic/{rel.as_posix()}"
        except ValueError:
            return None

    def _page_image_url(self, doc_source: str, page_number: int) -> Optional[str]:
        """Builds the HTTP URL for a full-page image from the document name and page number."""
        doc_stem = Path(doc_source).stem          # Remove .pdf suffix
        img_path = Path(settings.pic_dir) / doc_stem / f"page_{page_number}_img_0.png"
        if img_path.exists():
            rel = img_path.relative_to(Path(settings.pic_dir))
            return f"/static/pic/{rel.as_posix()}"
        return None

    def _build_citations(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[CitationSource]:
        """
        Builds fine-grained citation info for each chunk.
        Citation numbers start from 1, matching [Source X] markers in the LLM prompt.
        Figure chunks additionally carry image_url for frontend rendering.

        Auto-detection: if chunk text contains a "图X-X" reference and has page numbers,
        attaches the corresponding full-page image URL.
        """
        citations = []
        for i, chunk in enumerate(chunks):
            is_figure = bool(chunk.metadata.get("is_figure", False))
            image_path = chunk.metadata.get("image_path", "") or ""
            image_url = self._image_url(image_path) if is_figure else None

            # Chunk text contains "图X-X" reference → attach full-page image
            if not image_url and chunk.page_numbers and self._FIGURE_REF_RE.search(chunk.text):
                url = self._page_image_url(chunk.doc_source, chunk.page_numbers[0])
                if url:
                    image_url = url
                    is_figure = True

            citation = CitationSource(
                citation_id=i + 1,
                chunk_id=chunk.chunk_id,
                doc_source=chunk.doc_source or "Merck Manual",
                page_numbers=chunk.page_numbers,
                section_title=chunk.section_title,
                chapter=chunk.chapter,
                excerpt=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                relevance_score=round(chunk.rerank_score, 3),
                is_figure=is_figure,
                image_url=image_url,
            )
            citations.append(citation)
        return citations

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Formats the chunk list into a structured context string understandable by the LLM.

        Each chunk includes three header lines:
          ① Source number + warning marker + section hierarchy path + page numbers
          ② Medical entities involved (diseases/drugs/symptoms, helps LLM locate topic quickly)
          ③ Main text content
        """
        if not chunks:
            return "(No relevant reference materials found)"

        parts = []
        for i, chunk in enumerate(chunks):
            citation_num = i + 1
            is_figure = bool(chunk.metadata.get("is_figure", False))

            # ── ① Source line ───────────────────────────────────────────────
            warning = "【WARNING】" if chunk.metadata.get("has_warning") else ""

            location_parts = []
            if chunk.chapter:
                location_parts.append(chunk.chapter)
            if chunk.section_title and chunk.section_title != chunk.chapter:
                location_parts.append(chunk.section_title)
            location = " > ".join(location_parts) if location_parts else "《Merck Manual》"

            pages_str = ""
            if chunk.page_numbers:
                pages_str = f" (p.{', '.join(str(p) for p in chunk.page_numbers[:3])})"

            if is_figure:
                # ── Figure chunk: inform LLM this source is a figure, image already provided to user ──
                figure_caption = chunk.metadata.get("figure_caption", "") or chunk.text
                source_line = f"[Source {citation_num}] [Figure] {location}{pages_str}"
                block_parts = [
                    source_line,
                    f"  Caption: {figure_caption}",
                    "  (This source is a medical diagram. The image has been provided to the user separately. "
                    "Please cite [Source " + str(citation_num) + "] in your answer and prompt the user to view the figure.)",
                ]
            else:
                # ── Text chunk: standard logic ──────────────────────────────
                source_line = f"[Source {citation_num}] {warning}{location}{pages_str}"

                PRIORITY_TYPES = ["疾病", "药物", "症状", "实验室指标"]
                entity_segments = []
                for etype in PRIORITY_TYPES:
                    names = chunk.entities.get(etype, [])
                    if names:
                        entity_segments.append(f"{etype}: {', '.join(names[:3])}")
                entity_line = f"  [{' | '.join(entity_segments)}]" if entity_segments else ""

                block_parts = [source_line]
                if entity_line:
                    block_parts.append(entity_line)
                block_parts.append(chunk.text)

            parts.append("\n".join(block_parts))

        return "\n\n---\n\n".join(parts)
