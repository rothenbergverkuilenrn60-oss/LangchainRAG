"""
chunker.py — Multi-strategy document chunker
Implements three chunking strategies that work together:

  1. ParentChildChunker: first cuts large chunks (parent), then smaller chunks within parents (child)
     - Child chunks for precise vector retrieval
     - Parent chunks as context window passed to LLM

  2. SemanticChunker: detects semantic boundaries based on BGE-M3 embedding cosine similarity
     - Sharp drop in similarity between adjacent sentences → semantic breakpoint → chunk here
     - Avoids fixed-length splitting from breaking semantic integrity

  3. SummaryChunker: generates summaries for large parent chunks, stored separately as Summary vectors
     - Used for macro-level retrieval (first find relevant sections, then precisely locate)

Input: List[Dict] (page data output from PDFProcessor)
Output: List[DocumentChunk] (chunk list with complete metadata)
"""

import os
import re
import uuid
import asyncio
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

from app.models import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors"""
    dot = np.dot(a, b)                       # Dot product
    norm = np.linalg.norm(a) * np.linalg.norm(b)  # Product of norms
    return float(dot / norm) if norm > 0 else 0.0


class SemanticChunker:
    """
    Semantics-aware chunker.

    Core idea:
      1. Split text into sentences (preserving punctuation)
      2. Compute embedding similarity between adjacent sentences using a sliding window
      3. Positions where similarity drops below threshold → semantic breakpoints → chunk here
      4. Merge overly short segments into adjacent chunks to ensure minimum chunk size

    Args:
        embedding_model: sentence_transformers.SentenceTransformer instance
        breakpoint_threshold: cosine similarity below this triggers a split (default 0.45)
        min_chunk_size: minimum chunk character count (prevents fragmentation)
        max_chunk_size: maximum chunk character count (prevents excessive length)
    """

    # Chinese sentence splitting regex: split after sentence-ending punctuation, keep punctuation
    SENT_SPLIT_RE = re.compile(r'(?<=[。！？；\.\!\?;])\s*')

    def __init__(
        self,
        embedding_model,
        breakpoint_threshold: float = None,
        min_chunk_size: int = 100,
        max_chunk_size: int = None,
    ):
        self.model = embedding_model
        self.threshold = breakpoint_threshold or settings.semantic_breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size or settings.parent_chunk_size
        # Sentence embedding cache: {sentence_text: np.ndarray}, reused across split() calls
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def split(self, text: str) -> List[str]:
        """
        Semantically splits text into a list of chunks.

        Args:
            text: Input text to split

        Returns:
            List[str], each element is a semantic chunk
        """
        # Step 1: Split into sentences
        sentences = [s.strip() for s in self.SENT_SPLIT_RE.split(text) if s.strip()]
        if len(sentences) <= 2:
            # Too few sentences, not worth semantic splitting; return whole segment
            return [text] if len(text) > self.min_chunk_size else []

        # Step 2: Batch compute sentence embeddings (cache hits reused, reduces redundant encode calls)
        try:
            uncached = [s for s in sentences if s not in self._embedding_cache]
            if uncached:
                new_embs = self.model.encode(
                    uncached,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for sent, emb in zip(uncached, new_embs):
                    self._embedding_cache[sent] = emb
            embeddings = np.stack([self._embedding_cache[s] for s in sentences])
        except Exception as e:
            logger.warning(f"Semantic embedding computation failed, falling back to fixed-length splitting: {e}")
            return self._fixed_split(text)

        # Step 3: Compute adjacent sentence similarities, find breakpoints
        breakpoints = []
        # For 5 sentences at indices 0,1,2,3,4: i iterates 0,1,2,3, corresponding to pairs (0,1),(1,2),(2,3),(3,4)
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                # Sharp similarity drop → semantic boundary
                breakpoints.append(i + 1)  # Start a new chunk at sentence i+1

        # Step 4: Split sentence list at breakpoints
        '''
        chunks_sentences = [
            ["sentence A", "sentence B", "sentence C"],   # 1st semantic chunk (3 sentences)
            ["sentence D", "sentence E"],                  # 2nd semantic chunk (2 sentences)
            ["sentence F", "sentence G", "sentence H"],   # 3rd semantic chunk (3 sentences)
        ]
        '''
        chunks_sentences: List[List[str]] = []
        start = 0
        for bp in breakpoints:
            chunks_sentences.append(sentences[start:bp])
            start = bp
        chunks_sentences.append(sentences[start:])  # Last chunk

        # Step 5: Merge overly short chunks, limit overly long chunks
        chunks: List[str] = []  # Final list of text chunks to return
        buffer = ""             # Buffer for accumulating short chunks to avoid fragmentation
        for chunk_sents in chunks_sentences:
            chunk_text = "".join(chunk_sents)
            if len(buffer) + len(chunk_text) < self.min_chunk_size:
                # Too short, merge into buffer
                buffer += chunk_text
            elif len(chunk_text) > self.max_chunk_size:
                # Too long, split again with fixed length
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self._fixed_split(chunk_text))
            else:
                if buffer:
                    chunks.append(buffer)
                # Delay write; next chunk may be too short and need merging with current
                buffer = chunk_text

        if buffer:
            chunks.append(buffer)

        return [c for c in chunks if len(c) >= self.min_chunk_size]

    def _fixed_split(self, text: str) -> List[str]:
        """
        Fallback strategy: split by fixed character count (with overlap).
        Used when embedding computation fails or text is too long.
        """
        size = self.max_chunk_size
        overlap = settings.child_chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += size - overlap
        return chunks


class ParentChildChunker:
    """
    Parent-child chunker: splits documents into two granularity levels.

    - Parent Chunk: larger semantically complete paragraphs, stores original text for LLM
    - Child Chunk: smaller chunks cut from within parent chunks, for precise vector retrieval

    Retrieval: use child chunks for vector retrieval → find matching child → pass corresponding parent to LLM

    Args:
        embedding_model: embedding model for semantic splitting
        rule_engine: rule engine instance
        entity_extractor: entity extractor instance
    """

    def __init__(self, embedding_model, rule_engine, entity_extractor):
        self.semantic_chunker = SemanticChunker(embedding_model)
        self.rule_engine = rule_engine
        self.entity_extractor = entity_extractor

        # Child chunk fixed splitting parameters
        self._child_size = settings.child_chunk_size
        self._child_overlap = settings.child_chunk_overlap

    def create_chunks(
        self,
        pages: List[Dict[str, Any]],
        tenant_id: str,
        doc_source: str,
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Generates parent and child chunk lists from page data.

        Args:
            pages: Output of PDFProcessor.extract()
            tenant_id: Tenant ID
            doc_source: Source document filename

        Returns:
            (parent_chunks, child_chunks)
        """
        # ── Phase 1: Semantic splitting (serial, embedding model not thread-safe) ────
        # Concatenate all page texts but preserve page number mapping
        full_text_parts: List[Tuple[str, int, str, str]] = []
        for page in pages:
            if page["text"].strip():  # Skip empty pages
                full_text_parts.append((
                    page["text"],
                    page["page_number"],
                    page["chapter"],
                    page["section_title"],
                ))

        logger.info("Starting semantic splitting of parent chunks...")
        raw_items: List[Tuple[str, List[int], str, str]] = []
        accumulated = ""
        accumulated_pages: List[int] = []
        current_chapter = ""
        current_section = ""

        for text, page_num, chapter, section in full_text_parts:
            accumulated += text + "\n"
            accumulated_pages.append(page_num)
            if chapter:
                current_chapter = chapter
            if section:
                current_section = section

            if len(accumulated) >= settings.parent_chunk_size:
                parent_subs = self.semantic_chunker.split(accumulated)
                if not parent_subs:
                    parent_subs = [accumulated]
                for sub_text in parent_subs:
                    raw_items.append((sub_text, accumulated_pages.copy(), current_chapter, current_section))
                accumulated = ""
                accumulated_pages = []

        if accumulated.strip():
            parent_subs = self.semantic_chunker.split(accumulated)
            if not parent_subs:
                parent_subs = [accumulated]
            for sub_text in parent_subs:
                raw_items.append((sub_text, accumulated_pages.copy(), current_chapter, current_section))

        # ── Phase 2: Rule processing + entity extraction (parallel, both are thread-safe) ──
        def _process_item(item: Tuple[str, List[int], str, str]):
            sub_text, page_nums, chapter, section = item
            parent = self._make_parent_chunk(
                text=sub_text,
                page_numbers=page_nums,
                chapter=chapter,
                section_title=section,
                tenant_id=tenant_id,
                doc_source=doc_source,
            )
            return parent, self._make_child_chunks(parent)

        max_workers = min(4, os.cpu_count() or 1)
        parent_chunks: List[DocumentChunk] = []
        child_chunks: List[DocumentChunk] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for parent, children in executor.map(_process_item, raw_items):
                parent_chunks.append(parent)
                child_chunks.extend(children)

        # ── Phase 3: Figure chunks (generated from images field in pages) ────────
        figure_chunks = self._make_figure_chunks(pages, tenant_id, doc_source)
        child_chunks.extend(figure_chunks)

        logger.info(
            f"Chunking complete: {len(parent_chunks)} parent chunks, "
            f"{len(child_chunks)} child chunks (including {len(figure_chunks)} figure chunks)"
        )
        return parent_chunks, child_chunks

    def _make_parent_chunk(
        self,
        text: str,
        page_numbers: List[int],
        chapter: str,
        section_title: str,
        tenant_id: str,
        doc_source: str,
    ) -> DocumentChunk:
        """Creates a parent DocumentChunk with rule processing and entity recognition"""
        chunk_id = str(uuid.uuid4())  # Globally unique ID

        # Apply rule engine
        initial_meta = {
            "page_numbers": page_numbers,
            "chapter": chapter,
            "section_title": section_title,
            "tenant_id": tenant_id,
            "doc_source": doc_source,
            "chunk_type": "parent",
        }
        processed_text, enriched_meta = self.rule_engine.process(text, initial_meta)

        # Extract medical entities
        entities = self.entity_extractor.extract(processed_text)

        return DocumentChunk(
            chunk_id=chunk_id,
            text=processed_text,
            parent_chunk_id=None,  # Parent chunks have no parent
            page_numbers=sorted(set(page_numbers)),  # Deduplicate and sort
            section_title=section_title,
            chapter=chapter,
            tenant_id=tenant_id,
            doc_source=doc_source,
            entities=entities,
            metadata={
                **enriched_meta,
                "chunk_type": "parent",
                "char_count": len(processed_text),
            },
        )

    def _make_child_chunks(self, parent: DocumentChunk) -> List[DocumentChunk]:
        """
        Splits child chunks within a parent chunk using fixed size.
        Each child chunk's parent_chunk_id points to its parent, forming a hierarchy.
        """
        size = self._child_size
        overlap = self._child_overlap
        text = parent.text
        children: List[DocumentChunk] = []

        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            child_text = text[start:end]

            if len(child_text) < 20:  # Skip overly short child chunks
                break

            child_id = str(uuid.uuid4())
            children.append(DocumentChunk(
                chunk_id=child_id,
                text=child_text,
                parent_chunk_id=parent.chunk_id,  # Link to parent chunk
                page_numbers=parent.page_numbers,
                section_title=parent.section_title,
                chapter=parent.chapter,
                tenant_id=parent.tenant_id,
                doc_source=parent.doc_source,
                entities=parent.entities,  # Inherit parent entities
                metadata={
                    **parent.metadata,
                    "chunk_type": "child",
                    "parent_chunk_id": parent.chunk_id,
                    "char_count": len(child_text),
                },
            ))
            start += size - overlap  # Sliding window with overlap

        return children

    def _make_figure_chunks(
        self,
        pages: List[Dict[str, Any]],
        tenant_id: str,
        doc_source: str,
    ) -> List[DocumentChunk]:
        """
        Creates an independent child chunk (no parent) for each image extracted by PDFProcessor.

        Figure chunk text = caption (used for vector and BM25 retrieval).
        metadata stores image_path, is_figure, and other figure-specific fields.
        When a figure chunk is retrieved, context_processor attaches the image URL
        to the citation returned to the frontend.
        """
        figure_chunks: List[DocumentChunk] = []

        for page in pages:
            for img_info in page.get("images", []):
                image_path = img_info.get("image_path", "")
                if not image_path:
                    continue

                caption = img_info.get("caption", "")
                figure_number = img_info.get("figure_number", "")

                # Figure chunk retrieval text = caption; fallback description if no caption
                text = caption if caption else f"Figure (page {page['page_number']})"

                chunk_id = str(uuid.uuid4())
                figure_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    parent_chunk_id=None,      # Figure chunks have no parent
                    page_numbers=[page["page_number"]],
                    section_title=page.get("section_title", ""),
                    chapter=page.get("chapter", ""),
                    tenant_id=tenant_id,
                    doc_source=doc_source,
                    entities={},
                    metadata={
                        "chunk_type": "figure",
                        "is_figure": True,
                        "image_path": image_path,
                        "figure_caption": caption,
                        "figure_number": figure_number,
                        "char_count": len(text),
                        "is_low_quality": False,
                        "has_warning": False,
                        "has_table": False,
                        "importance_boost": 1.0,
                    },
                ))

        return figure_chunks


class SummaryChunker:
    """
    Summary index generator: calls LLM to generate summaries for parent chunks,
    stored in a separate summaries collection.

    Purpose:
      Macro-level retrieval — first find relevant sections using summary vectors,
      then precisely retrieve child chunks within those sections.
      Effectively solves the "needle in a haystack" problem (especially when query
      and child chunk semantics diverge significantly).

    Design:
      - Async concurrent LLM calls (asyncio.Semaphore controls concurrency, avoids API rate limits)
      - Exponential backoff retry (up to max_retries times, covers timeouts/transient failures)
      - generate_summaries() provides sync entry point for ingest scripts and thread pools

    Args:
        llm: LangChain ChatModel instance (must support ainvoke)
        min_chunk_len: parent chunks shorter than this are skipped for summary generation
        max_concurrent: max concurrent LLM calls (5 recommended, ~300 RPM for DeepSeek)
        max_retries: max retry count per chunk
    """

    SUMMARY_PROMPT = (
        "请用1-2句话概括以下医学文本的核心内容，"
        "突出主要疾病、症状或治疗要点，不要包含无关信息：\n\n{text}\n\n摘要："
    )

    def __init__(
        self,
        llm,
        min_chunk_len: int = 200,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ):
        self.llm = llm
        self.min_chunk_len = min_chunk_len
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

    def generate_summaries(
        self,
        parent_chunks: List[DocumentChunk],
    ) -> List[Dict[str, Any]]:
        """
        Sync entry point (for ingest scripts and asyncio.to_thread calls).
        Internally uses asyncio.run() for concurrent execution, ~max_concurrent times faster than serial.
        """
        return asyncio.run(self.generate_summaries_async(parent_chunks))

    async def generate_summaries_async(
        self,
        parent_chunks: List[DocumentChunk],
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously generates summaries in batch with concurrency control and retries.

        Args:
            parent_chunks: Parent chunk list (from ParentChildChunker.create_chunks)

        Returns:
            List[Dict], format compatible with ChromaIndexer.index_summaries() input:
              {"chunk_id": str, "text": str, "metadata": dict}
        """
        candidates = [c for c in parent_chunks if len(c.text) >= self.min_chunk_len]
        total = len(candidates)
        logger.info(f"Starting concurrent summary generation: {total} parent chunks (concurrency={self.max_concurrent})")

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._generate_one(chunk, semaphore, idx, total)
            for idx, chunk in enumerate(candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = [r for r in results if isinstance(r, dict)]
        failed = sum(1 for r in results if isinstance(r, Exception))
        if failed:
            logger.warning(f"Summary generation failed: {failed}/{total} chunks (skipped)")
        logger.info(f"Summary generation complete: {len(summaries)}/{total} parent chunks")
        return summaries

    async def _generate_one(
        self,
        chunk: DocumentChunk,
        semaphore: asyncio.Semaphore,
        idx: int,
        total: int,
    ) -> Dict[str, Any]:
        """
        Generates a summary for a single parent chunk with semaphore rate limiting
        and exponential backoff retry.
        If it fails more than max_retries times, raises an exception
        (caught by gather as an Exception instance).
        """
        from langchain_core.messages import HumanMessage

        async with semaphore:
            last_exc: Exception = RuntimeError("Not executed")
            for attempt in range(self.max_retries):
                try:
                    prompt = self.SUMMARY_PROMPT.format(text=chunk.text[:1200])
                    response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                    summary_text = response.content.strip()
                    if not summary_text:
                        raise ValueError("LLM returned empty summary")
                    if (idx + 1) % 50 == 0:
                        logger.info(f"  Summary progress: {idx + 1}/{total}")
                    return {
                        "chunk_id": f"summary_{chunk.chunk_id}",
                        "text": summary_text,
                        "metadata": {
                            "source_chunk_id": chunk.chunk_id,
                            "page_numbers": chunk.page_numbers,
                            "chapter": chunk.chapter,
                            "section_title": chunk.section_title,
                            "tenant_id": chunk.tenant_id,
                            "doc_source": chunk.doc_source,
                            "chunk_type": "summary",
                        },
                    }
                except Exception as e:
                    last_exc = e
                    if attempt < self.max_retries - 1:
                        wait = 2 ** attempt  # 1s → 2s → 4s
                        logger.warning(
                            f"Summary generation failed chunk={chunk.chunk_id[:8]} "
                            f"(attempt {attempt + 1}/{self.max_retries}): {e}, retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
            raise last_exc
