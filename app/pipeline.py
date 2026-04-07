"""
pipeline.py — RAG main pipeline
Connects all modules into a complete end-to-end RAG workflow:

Full data flow:
  User question
    ↓ QueryProcessor (rewrite + HyDE + Multi-Query)
    ↓ HybridRetriever (BM25 + Vector, parallel per query variant)
    ↓ DualRRFFusion (Layer1: per-query hybrid, Layer2: cross-query aggregation)
    ↓ Deduplication (at chunk_id level)
    ↓ BGEReranker (precise cross-encoder scoring)
    ↓ ContextProcessor (compression + reordering + citation building)
    ↓ RAGGenerator (streaming generation + Self-RAG verification)
    ↓ Streaming SSE output → client

All components are lazily loaded on startup (initialized on first request)
to avoid occupying too many resources at startup.
Module-level singleton caches component instances for reuse across requests
(saves model loading time).
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import (
    QueryRequest, RAGResponse, StreamEvent, QueryMode,
    RetrievedChunk, CitationSource
)
from app.preprocessing.entity_extractor import MedicalEntityExtractor
from app.indexing.indexer import ChromaIndexer
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import BGEReranker
from app.generation.context_processor import ContextProcessor
from app.generation.generator import RAGGenerator
from app.tenants.manager import tenant_manager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Component container (lazily initialized singleton)
# ─────────────────────────────────────────────────────────────────────────────

class ComponentContainer:
    """
    Lazily initialized component container.
    All expensive components (model loading) are initialized only on first use,
    then reused. asyncio.Lock ensures no duplicate initialization under concurrency.
    """

    def __init__(self):
        self._embedding_model: Optional[SentenceTransformer] = None
        self._indexer: Optional[ChromaIndexer] = None
        self._query_processor: Optional[QueryProcessor] = None
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[BGEReranker] = None
        self._context_processor: Optional[ContextProcessor] = None
        self._generator: Optional[RAGGenerator] = None
        self._entity_extractor: Optional[MedicalEntityExtractor] = None
        self._init_lock = asyncio.Lock()  # Prevents duplicate initialization under concurrency
        self._initialized = False

    async def initialize(self):
        """
        Asynchronously initialize all components.
        Uses asyncio.Lock to guarantee single initialization (even with concurrent requests).
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:  # Double-checked locking
                return

            logger.info("Initializing RAG components...")

            # 1. Load BGE-M3 Embedding model (most time-consuming, ~5-10s)
            logger.info(f"Loading embedding model: {settings.embedding_model_path}")
            self._embedding_model = await asyncio.to_thread(
                SentenceTransformer,
                settings.embedding_model_path,
                device="cuda",  # Prefer GPU
            )

            # 2. Initialize Chroma indexer (includes BM25 management)
            self._indexer = await asyncio.to_thread(
                ChromaIndexer,
                self._embedding_model,
            )

            # 3. Initialize retrieval components
            self._query_processor = QueryProcessor()          # LLM query expansion
            self._retriever = HybridRetriever(self._indexer)  # BM25 + vector

            # 4. Initialize Reranker (BGE-Reranker-v2-M3)
            self._reranker = BGEReranker()
            # Trigger model loading (avoids delay on first request)
            await asyncio.to_thread(self._reranker._load_model)

            # 5. Initialize generation components
            self._context_processor = ContextProcessor()
            self._generator = RAGGenerator()

            # 6. Entity extractor
            self._entity_extractor = MedicalEntityExtractor()

            self._initialized = True
            logger.info("All RAG components initialized successfully")

    @property
    def embedding_model(self) -> SentenceTransformer:
        self._assert_initialized()
        return self._embedding_model

    @property
    def indexer(self) -> ChromaIndexer:
        self._assert_initialized()
        return self._indexer

    @property
    def query_processor(self) -> QueryProcessor:
        self._assert_initialized()
        return self._query_processor

    @property
    def retriever(self) -> HybridRetriever:
        self._assert_initialized()
        return self._retriever

    @property
    def reranker(self) -> BGEReranker:
        self._assert_initialized()
        return self._reranker

    @property
    def context_processor(self) -> ContextProcessor:
        self._assert_initialized()
        return self._context_processor

    @property
    def generator(self) -> RAGGenerator:
        self._assert_initialized()
        return self._generator

    @property
    def entity_extractor(self) -> MedicalEntityExtractor:
        self._assert_initialized()
        return self._entity_extractor

    def _assert_initialized(self):
        if not self._initialized:
            raise RuntimeError("Components not initialized. Call `await container.initialize()` first.")


# Global component container singleton
_container = ComponentContainer()


async def get_container() -> ComponentContainer:
    """FastAPI dependency injection function: returns the initialized component container"""
    await _container.initialize()
    return _container


# ─────────────────────────────────────────────────────────────────────────────
# RAG main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.
    A new Pipeline instance is created per request (lightweight, holds references only).

    Usage:
        pipeline = RAGPipeline(container)

        # Streaming output
        async for event in pipeline.query_stream(request):
            yield event

        # Non-streaming output
        response = await pipeline.query(request)
    """

    def __init__(self, container: ComponentContainer):
        self.container = container

    async def query_stream(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[StreamEvent]:
        """
        Full streaming RAG query workflow.

        Args:
            request: QueryRequest object

        Yields:
            StreamEvent: stream events (token / citations / metadata / done / error)
        """
        tenant_id = request.tenant_id
        question = request.question
        mode = request.mode
        filters = request.filters
        top_n = request.top_n or settings.rerank_top_n

        # Ensure tenant is registered
        tenant_manager.get_or_create(tenant_id)

        try:
            # ── Step 0: Check if retrieval is needed ──────────────────────────
            needs_retrieval = await self.container.generator.check_retrieval_needed(question)
            if not needs_retrieval:
                async for event in self.container.generator.stream_direct(question):
                    yield event
                return

            # ── Step 1: Query expansion ───────────────────────────────────────
            yield StreamEvent(
                event="metadata",
                data={"status": "expanding_query", "message": "Expanding query..."},
            )
            query_variants_dict = await self.container.query_processor.expand(
                question=question,
                mode=mode,
            )
            all_queries = query_variants_dict["all"]
            logger.info(f"Query expansion: {len(all_queries)} variants")

            # ── Step 2: Hybrid retrieval (parallel dual-lane + dual-layer RRF) ──
            yield StreamEvent(
                event="metadata",
                data={"status": "retrieving", "message": "Retrieving relevant documents..."},
            )

            # Convert query variants list to dict (keyed by variant name for RRF tracking)
            query_variants_map: Dict[str, str] = {}
            for i, q in enumerate(all_queries):
                if i == 0:
                    key = "original"
                elif i == 1:
                    key = "rewritten"
                elif i == 2:
                    key = "hyde"
                else:
                    key = f"multi_q{i-2}"
                query_variants_map[key] = q

            # Merge request filters with tenant default filters
            effective_cfg = tenant_manager.get_effective_config(tenant_id)
            # Get default_filters from config (default empty dict), copy it as initial filters
            merged_filters = {**effective_cfg.get("default_filters", {})}
            if filters:
                merged_filters.update(filters)

            raw_docs = await self.container.retriever.retrieve(
                query_variants=query_variants_map,
                tenant_id=tenant_id,
                filters=merged_filters if merged_filters else None,
                top_n=top_n * 2,  # Fetch extra candidates for Reranker to select from
            )

            if not raw_docs:
                yield StreamEvent(
                    event="error",
                    data={"message": "No relevant documents found. Please ensure documents have been ingested (/ingest)."},
                )
                return

            # ── Step 3: Reranker fine-ranking ─────────────────────────────────
            yield StreamEvent(
                event="metadata",
                data={"status": "reranking", "message": "Re-ranking results..."},
            )
            reranked_chunks = await asyncio.to_thread(
                self.container.reranker.rerank,
                query=question,
                docs=raw_docs,
                top_n=top_n,
            )

            # ── Step 4: Context processing (compression + reordering + citation building) ──
            context_str, citations = self.container.context_processor.process(
                chunks=reranked_chunks,
                query=question,
            )

            # ── Step 5: Entity extraction (from query, for metadata output) ────
            query_entities = self.container.entity_extractor.extract_from_query(question)

            # Push retrieval summary (let user know what materials were found)
            yield StreamEvent(
                event="metadata",
                data={
                    "status": "generating",
                    "message": "Generating answer...",
                    "retrieved_count": len(reranked_chunks),
                    "query_variants": all_queries,
                    "entities_in_query": query_entities,
                },
            )

            # ── Step 6: Streaming generation + Self-RAG ───────────────────────
            async for event in self.container.generator.stream(
                question=question,
                context=context_str,
                citations=citations,
                chunks=reranked_chunks,
            ):
                yield event

            # Update tenant query statistics
            tenant_manager.record_query(tenant_id)

        except Exception as e:
            logger.exception(f"RAG pipeline error: {e}")
            yield StreamEvent(
                event="error",
                data={"message": f"Processing error: {str(e)}"},
            )

    async def query(self, request: QueryRequest) -> RAGResponse:
        """
        Non-streaming query (collects all stream events and aggregates into a complete response).
        """
        tenant_id = request.tenant_id
        question = request.question
        mode = request.mode
        top_n = request.top_n or settings.rerank_top_n

        tenant_manager.get_or_create(tenant_id)

        # Step 0: Check if retrieval is needed
        needs_retrieval = await self.container.generator.check_retrieval_needed(question)
        if not needs_retrieval:
            import time
            start_time = time.time()
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_core.output_parsers import StrOutputParser
            messages = [
                SystemMessage(content="你是一个知识渊博的助手，请直接、准确地回答用户问题。"),
                HumanMessage(content=question),
            ]
            response = await self.container.generator._llm.ainvoke(messages)
            answer = StrOutputParser().invoke(response)
            return RAGResponse(
                answer=answer,
                citations=[],
                retrieved_chunks=[],
                query_variants=[question],
                entities_in_query={},
                self_rag_passed=True,
                metadata={"elapsed_seconds": round(time.time() - start_time, 2), "retrieval_skipped": True},
            )

        # Query expansion
        query_variants_dict = await self.container.query_processor.expand(question, mode)
        all_queries = query_variants_dict["all"]

        query_variants_map = {
            (["original", "rewritten", "hyde"] + [f"multi_q{i}" for i in range(10)])[i]: q
            for i, q in enumerate(all_queries)
        }

        effective_cfg = tenant_manager.get_effective_config(tenant_id)
        merged_filters = {**effective_cfg.get("default_filters", {})}
        if request.filters:
            merged_filters.update(request.filters)

        raw_docs = await self.container.retriever.retrieve(
            query_variants=query_variants_map,
            tenant_id=tenant_id,
            filters=merged_filters if merged_filters else None,
            top_n=top_n * 5,
        )

        reranked_chunks = await asyncio.to_thread(
            self.container.reranker.rerank,
            query=question,
            docs=raw_docs,
            top_n=top_n,
        )

        context_str, citations = self.container.context_processor.process(
            chunks=reranked_chunks,
            query=question,
        )

        query_entities = self.container.entity_extractor.extract_from_query(question)

        response = await self.container.generator.generate(
            question=question,
            context=context_str,
            citations=citations,
            chunks=reranked_chunks,
        )

        # Attach pipeline-level metadata
        response.query_variants = all_queries
        response.entities_in_query = query_entities

        # Extract image info (for non-streaming clients to get image list directly)
        response.images = [
            {
                "citation_id": c.citation_id,
                "image_url": c.image_url,
                "caption": c.excerpt,
                "page_numbers": c.page_numbers,
            }
            for c in response.citations
            if c.is_figure and c.image_url
        ]

        tenant_manager.record_query(tenant_id)
        return response
