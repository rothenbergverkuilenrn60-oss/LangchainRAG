"""
models.py — Pydantic data model definitions
Defines all structured models for API requests/responses and internal data flow.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class QueryMode(str, Enum):
    """Query mode enum: controls whether Multi-Query / HyDE are enabled"""
    full = "full"           # Enable all: multi-query + rewrite + HyDE
    rewrite_only = "rewrite_only"  # Only rewrite + original query
    basic = "basic"         # Only original query (fast mode)


# ─────────────────────────────────────────────────────────────────────────────
# Internal data models (passed through the pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """
    A single document chunk: the smallest retrieval unit after splitting and preprocessing.
    Each chunk contains text content + rich metadata for tracing and filtering.
    """
    chunk_id: str                         # Unique identifier (UUID)
    text: str                             # Chunk text content
    parent_chunk_id: Optional[str] = None # Parent chunk ID (for Parent-Child indexing)
    page_numbers: List[int] = []          # Corresponding PDF page numbers (may span pages)
    section_title: str = ""               # Section title the chunk belongs to
    chapter: str = ""                     # Chapter the chunk belongs to
    tenant_id: str = "default"            # Owning tenant
    doc_source: str = ""                  # Source document filename
    entities: Dict[str, List[str]] = {}   # Recognized entities {type: [entity names]}
    summary: Optional[str] = None         # Chunk summary (for Summary Indexing)
    metadata: Dict[str, Any] = {}         # Extended metadata (for Chroma filtering)


class RetrievedChunk(BaseModel):
    """
    A retrieved chunk: extends DocumentChunk with retrieval score information.
    """
    chunk_id: str
    text: str
    score: float = 0.0        # Final RRF / Rerank score (higher = more relevant)
    bm25_rank: int = 0        # Rank in BM25 results (0 = not present)
    vector_rank: int = 0      # Rank in vector retrieval results (0 = not present)
    rrf_score: float = 0.0    # RRF fusion score
    rerank_score: float = 0.0 # Reranker score
    page_numbers: List[int] = []
    section_title: str = ""
    chapter: str = ""
    doc_source: str = ""
    entities: Dict[str, List[str]] = {}
    metadata: Dict[str, Any] = {}


class CitationSource(BaseModel):
    """
    Fine-grained citation structure: source information for each citation in the final answer.
    """
    citation_id: int                    # Citation number, corresponds to [1], [2] markers in text
    chunk_id: str                       # Source chunk ID
    doc_source: str                     # Source document filename
    page_numbers: List[int]             # Page numbers
    section_title: str                  # Section title
    chapter: str                        # Chapter name
    excerpt: str                        # Original text excerpt (first 200 chars)
    relevance_score: float              # Relevance score relative to the query
    is_figure: bool = False             # Whether this is an image source
    image_url: Optional[str] = None    # Image HTTP URL (only when is_figure=True)


# ─────────────────────────────────────────────────────────────────────────────
# API request models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    Request body for the /query endpoint.
    """
    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    tenant_id: str = Field(default="default", description="Tenant ID (multi-tenant isolation)")
    mode: QueryMode = Field(default=QueryMode.full, description="Query mode")
    # Metadata filters: e.g. {"chapter": "Cardiology"}
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")
    top_n: Optional[int] = Field(default=None, description="Override default rerank_top_n")
    stream: bool = Field(default=True, description="Whether to enable streaming output")


class IngestRequest(BaseModel):
    """
    Request body for the /ingest endpoint (triggers document ingestion).
    """
    tenant_id: str = Field(default="default", description="Tenant ID")
    pdf_path: Optional[str] = Field(default=None, description="PDF path (uses default if not set)")
    force_reindex: bool = Field(default=False, description="Force rebuild index")
    with_summary: bool = Field(default=False, description="Generate summary index (requires LLM API, takes longer)")


# ─────────────────────────────────────────────────────────────────────────────
# API response models (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """
    Complete response for a non-streaming query.
    """
    answer: str                             # Final answer generated by LLM
    citations: List[CitationSource]         # Fine-grained citation list
    retrieved_chunks: List[RetrievedChunk]  # Retrieved chunks used for generation
    query_variants: List[str]               # Actual query variants used (including rewrite/HyDE)
    entities_in_query: Dict[str, List[str]] # Entities recognized from the question
    self_rag_passed: bool = True            # Whether Self-RAG verification passed
    metadata: Dict[str, Any] = {}          # Extra metadata (elapsed time, token count, etc.)
    images: List[Dict[str, Any]] = []       # Retrieved images list (with url, caption, citation_id)


class IngestResponse(BaseModel):
    """
    Response for the /ingest endpoint.
    """
    status: str
    tenant_id: str
    chunks_indexed: int
    collection_name: str
    message: str


class HealthResponse(BaseModel):
    """
    Response for the /health endpoint.
    """
    status: str = "ok"
    chroma_connected: bool = False
    embedding_model_loaded: bool = False
    reranker_loaded: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Streaming event models
# ─────────────────────────────────────────────────────────────────────────────

class StreamEvent(BaseModel):
    """
    Data format for each event in an SSE (Server-Sent Events) stream.
    The frontend distinguishes token increments vs full citation info via the event field.
    """
    event: str           # Event type: "token" | "citations" | "metadata" | "done" | "error"
    data: Any            # Event payload
