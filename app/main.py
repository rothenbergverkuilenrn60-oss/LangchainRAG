"""
main.py — FastAPI application entry point
Defines all HTTP endpoints with streaming SSE output support.

Endpoint list:
  GET  /health          — Health check (component status)
  POST /ingest          — Document ingestion (PDF → vectorization → storage)
  POST /query           — RAG query (streaming SSE or full JSON)
  GET  /tenants         — List all tenants
  GET  /tenants/{id}    — Get tenant config and stats
  POST /tenants/{id}/reset — Reset tenant data (dangerous operation)

Streaming output format (SSE):
  Each event is sent as "data: {...}\n\n"
  Frontend receives via EventSource or fetch + ReadableStream
"""

import json
import pickle
import hashlib
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    QueryRequest, QueryMode, IngestRequest, IngestResponse,
    HealthResponse, RAGResponse, StreamEvent
)
from app.pipeline import get_container, ComponentContainer, RAGPipeline
from app.tenants.manager import tenant_manager

# ─────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan event handler
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management: preload components on startup, cleanup on shutdown."""
    # ── Startup phase ──
    logger.info("=" * 60)
    logger.info("Medical RAG API starting...")
    logger.info(f"Chroma DB path: {settings.chroma_db_path}")
    logger.info(f"Default tenant: {settings.default_tenant_id}")
    logger.info("=" * 60)
    asyncio.create_task(get_container())

    yield  # Application running normally

    # ── Shutdown phase ──
    logger.info("Medical RAG API shutting down...")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical RAG API",
    description="Advanced RAG medical Q&A system based on the Merck Manual",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI URL
    redoc_url="/redoc",     # ReDoc documentation URL
    lifespan=lifespan,      # Register lifecycle handler
)

# ── Middleware configuration ──────────────────────────────────────────────────

# CORS: allow all origins (restrict to specific domains in production)
app.add_middleware(
    CORSMiddleware,             # Cross-Origin Resource Sharing middleware
    allow_origins=["*"],        # Allow all origins
    allow_credentials=True,     # Allow cookies/auth headers
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all request headers
)

# Gzip compression: reduces response size (especially effective for large JSON responses)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Static file serving: mount image directory (images extracted during ingest are accessed via this path)
# URL: /static/pic/{pdf_stem}/page_{n}_img_{i}.png
_pic_dir = Path(settings.pic_dir)
_pic_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/pic", StaticFiles(directory=str(_pic_dir)), name="static_pic")


# ─────────────────────────────────────────────────────────────────────────────
# SSE stream utility function
# ─────────────────────────────────────────────────────────────────────────────

async def event_stream_generator(request: QueryRequest, container: ComponentContainer):
    """
    Converts RAGPipeline.query_stream() StreamEvents into SSE-format byte stream.

    SSE format:
      data: {"event": "token", "data": {"token": "hypertension"}}\n\n
      data: {"event": "citations", "data": {...}}\n\n
      data: {"event": "done", "data": {"message": "Generation complete"}}\n\n

    Frontend example (JavaScript):
      const es = new EventSource('/query-stream');
      es.onmessage = (e) => { const ev = JSON.parse(e.data); ... };
    """
    pipeline = RAGPipeline(container)
    async for event in pipeline.query_stream(request):
        # Serialize to JSON and format as SSE message
        payload = json.dumps(
            {"event": event.event, "data": event.data},
            ensure_ascii=False,  # Keep Chinese characters, don't escape to \uXXXX
        )
        # SSE format requires each message to start with "data: " and end with "\n\n"
        yield f"data: {payload}\n\n".encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Checks the loading status of each component, for use by load balancers and monitoring systems.
    """
    try:
        container = await get_container()
        # Try to get Chroma collection count to verify connection
        stats = container.indexer.get_collection_stats(settings.default_tenant_id)
        chroma_ok = True
    except Exception:
        chroma_ok = False

    return HealthResponse(
        status="ok",
        chroma_connected=chroma_ok,
        embedding_model_loaded=True,
        reranker_loaded=True,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    container: ComponentContainer = Depends(get_container),
):
    """
    Document ingestion endpoint: vectorizes PDF documents and stores them in Chroma + BM25 index.

    - Supports multi-tenancy: different tenant_ids store to separate vector collections
    - Supports force rebuild: force_reindex=true clears and rebuilds
    - Ingestion is time-consuming for large documents (may take several minutes), async recommended

    Note: First-time ingestion of large documents (e.g., Merck Manual) takes ~10-30 minutes.
    """
    tenant_id = request.tenant_id
    pdf_path = request.pdf_path or str(
        Path(settings.pdf_folder) / settings.pdf_filename
    )

    # Validate file exists
    if not Path(pdf_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {pdf_path}"
        )

    # Register tenant
    tenant_manager.get_or_create(tenant_id, description="Registered via API")

    # If force rebuild, clear tenant data first
    if request.force_reindex:
        logger.info(f"Force rebuilding index: tenant={tenant_id}")
        await asyncio.to_thread(container.indexer.reset_tenant, tenant_id)

    # Check if index already exists
    stats = await asyncio.to_thread(
        container.indexer.get_collection_stats, tenant_id
    )
    if stats.get("children", 0) > 0 and not request.force_reindex:
        return IngestResponse(
            status="skipped",
            tenant_id=tenant_id,
            chunks_indexed=stats["children"],
            collection_name=f"{settings.collection_prefix}_{tenant_id}_children",
            message=f"Index already exists ({stats['children']} child chunks). Skipping ingestion. Set force_reindex=true to rebuild.",
        )

    # Execute ingestion (in thread pool to avoid blocking the event loop)
    chunks_count = await asyncio.to_thread(
        _run_ingest_sync,
        pdf_path=pdf_path,
        tenant_id=tenant_id,
        container=container,
        with_summary=request.with_summary,
    )

    tenant_manager.record_ingest(tenant_id, chunks_count)

    return IngestResponse(
        status="success",
        tenant_id=tenant_id,
        chunks_indexed=chunks_count,
        collection_name=f"{settings.collection_prefix}_{tenant_id}_children",
        message=f"Successfully ingested {chunks_count} chunks",
    )


def _run_ingest_sync(
    pdf_path: str,
    tenant_id: str,
    container: ComponentContainer,
    with_summary: bool = False,
) -> int:
    """
    Internal synchronous function for document ingestion (runs in thread pool).
    Encapsulates time-consuming PDF processing, chunking, and embedding computation.
    """
    from app.preprocessing.pdf_processor import PDFProcessor
    from app.preprocessing.rule_engine import MedicalRuleEngine
    from app.preprocessing.entity_extractor import MedicalEntityExtractor
    from app.indexing.chunker import ParentChildChunker

    logger.info(f"Starting document ingestion: {pdf_path} → tenant={tenant_id}")

    pdf_file = Path(pdf_path)

    # Chunk result cache: MD5 fingerprint based on PDF file stat + chunking parameters
    stat = pdf_file.stat()
    cache_key = hashlib.md5(
        f"{stat.st_size}_{stat.st_mtime}_{settings.parent_chunk_size}_"
        f"{settings.child_chunk_size}_{settings.semantic_breakpoint_threshold}".encode()
    ).hexdigest()
    cache_dir = Path(settings.bm25_index_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_cache_path = cache_dir / f"chunks_{cache_key}.pkl"

    if chunk_cache_path.exists():
        # Cache hit: load directly, skip PDF extraction and embedding computation
        logger.info("Chunk cache detected, loading directly (skipping PDF extraction and embedding)...")
        with open(chunk_cache_path, "rb") as f:
            parent_chunks, child_chunks = pickle.load(f)
        logger.info(f"Cache loaded: {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")
    else:
        # 1. PDF extraction
        processor = PDFProcessor(pdf_path)
        pages = processor.extract()

        # 2. Initialize rule engine and entity extractor
        rule_engine = MedicalRuleEngine()
        entity_extractor = MedicalEntityExtractor()

        # 3. Chunking (parent-child + semantic chunking)
        chunker = ParentChildChunker(
            embedding_model=container.embedding_model,
            rule_engine=rule_engine,
            entity_extractor=entity_extractor,
        )
        parent_chunks, child_chunks = chunker.create_chunks(
            pages=pages,
            tenant_id=tenant_id,
            doc_source=pdf_file.name,
        )
        # Persist chunk results for reuse next time
        with open(chunk_cache_path, "wb") as f:
            pickle.dump((parent_chunks, child_chunks), f)
        logger.info(f"Chunk results cached: {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")

    # 4. Write to Chroma + build BM25 index
    total = container.indexer.index_chunks(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        tenant_id=tenant_id,
    )

    # 5. Generate summary index (optional)
    if with_summary:
        from langchain_deepseek import ChatDeepSeek
        from app.indexing.chunker import SummaryChunker
        logger.info("Generating summary index...")
        summary_llm = ChatDeepSeek(
            model=settings.llm_model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.1,
            max_tokens=150,
            streaming=False,
        )
        summary_chunker = SummaryChunker(summary_llm)
        summaries = summary_chunker.generate_summaries(parent_chunks)
        container.indexer.index_summaries(summaries, tenant_id=tenant_id)
        logger.info(f"Summary index complete: {len(summaries)} summary chunks")

    logger.info(f"Ingestion complete: {total} chunks written to index")
    return total


@app.post("/query", tags=["Q&A"])
async def query(
    request: QueryRequest,
    container: ComponentContainer = Depends(get_container),
):
    """
    RAG query endpoint (supports both streaming and non-streaming modes).

    **Streaming mode** (stream=true, recommended):
    Returns SSE stream, pushing the answer token by token, with citation info and metadata.

    SSE event types:
    - `token`: incremental text token (frontend appends to display)
    - `citations`: complete citation info (sent after answer generation)
    - `metadata`: statistics (elapsed time, Self-RAG result, etc.)
    - `done`: stream end marker
    - `error`: error information

    **Non-streaming mode** (stream=false):
    Returns a complete JSON response (answer, citations, retrieved chunk list).
    Suitable for batch processing or clients that don't support SSE.

    Example request:
    ```json
    {
      "question": "What are the first-line drugs for hypertension?",
      "tenant_id": "default",
      "mode": "full",
      "stream": true
    }
    ```
    """
    # Validate tenant (auto-create new tenant)
    tenant_manager.get_or_create(request.tenant_id)

    if request.stream:
        # Streaming response: text/event-stream format
        return StreamingResponse(
            event_stream_generator(request, container),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",         # Disable caching
                "Connection": "keep-alive",           # Keep connection alive
                "X-Accel-Buffering": "no",           # Disable Nginx buffering (important!)
                "Transfer-Encoding": "chunked",       # Chunked transfer encoding
            },
        )
    else:
        # Non-streaming response: complete JSON
        pipeline = RAGPipeline(container)
        response = await pipeline.query(request)
        return JSONResponse(content=response.model_dump())


@app.get("/tenants", tags=["Multi-tenancy"])
async def list_tenants():
    """List all active tenants and their statistics"""
    tenants = tenant_manager.list_tenants()
    return {
        "total": len(tenants),
        "tenants": [t.model_dump() for t in tenants],
    }


@app.get("/tenants/{tenant_id}", tags=["Multi-tenancy"])
async def get_tenant(
    tenant_id: str,
    container: ComponentContainer = Depends(get_container),
):
    """Get detailed information for a specified tenant (including index statistics)"""
    config = tenant_manager.get(tenant_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    stats = await asyncio.to_thread(
        container.indexer.get_collection_stats, tenant_id
    )

    return {
        "config": config.model_dump(),
        "index_stats": stats,
    }


@app.post("/tenants/{tenant_id}/reset", tags=["Multi-tenancy"])
async def reset_tenant(
    tenant_id: str,
    container: ComponentContainer = Depends(get_container),
):
    """
    Dangerous operation: clears all vector index and BM25 data for a specified tenant.
    This operation is irreversible. /ingest must be re-run to restore.
    """
    if not tenant_manager.exists(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    await asyncio.to_thread(container.indexer.reset_tenant, tenant_id)
    logger.warning(f"Tenant data reset: {tenant_id}")

    return {"status": "success", "message": f"Index data for tenant {tenant_id} has been cleared"}


@app.get("/", tags=["System"])
async def root():
    """API root path, returns basic information"""
    return {
        "name": "Medical RAG API",
        "version": "1.0.0",
        "description": "Advanced RAG medical Q&A system based on the Merck Manual",
        "docs": "/docs",
        "health": "/health",
    }
