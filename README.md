# LangchainRAG

A production-grade Retrieval-Augmented Generation (RAG) system built for medical document Q&A. Implements a multi-stage pipeline with hybrid retrieval, neural reranking, and streaming generation with Self-RAG verification.

## Architecture

```
User Question
    ↓ QueryProcessor    — query rewriting + HyDE + Multi-Query (3 variants)
    ↓ HybridRetriever   — BM25 (keyword) + Vector (semantic), parallel per variant
    ↓ DualRRFFusion     — Layer 1: per-query hybrid fusion
                          Layer 2: cross-query aggregation
    ↓ Deduplication     — chunk-level dedup
    ↓ BGEReranker       — cross-encoder precise scoring
    ↓ ContextProcessor  — compression + reordering + citation building
    ↓ RAGGenerator      — streaming generation + Self-RAG verification
    ↓ SSE stream → client
```

## Features

- **Hybrid retrieval** — BM25 keyword matching + BGE-M3 vector search, fused via dual-layer RRF
- **Multi-Query expansion** — rewrites each query into 3 variants (original, rewritten, HyDE) for higher recall
- **Summary-guided retrieval** — section summaries guide coarse retrieval before precise child-chunk lookup
- **Neural reranking** — BGE-reranker-v2-m3 cross-encoder scores all candidates before generation
- **Self-RAG verification** — generated answers are checked against retrieved context
- **Streaming SSE** — token-by-token streaming to the client
- **Multi-tenancy** — isolated ChromaDB collections per tenant (`{prefix}_{tenant_id}`)
- **Lazy initialization** — heavy models load only on first request; singletons reused across requests

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding | BAAI/bge-m3 (local) |
| Reranker | BAAI/bge-reranker-v2-m3 (local) |
| Vector DB | ChromaDB |
| Keyword search | BM25 |
| LLM | DeepSeek Chat API |
| PDF processing | Custom chunker (parent/child + semantic) |
| Entity extraction | MedicalEntityExtractor |
| API server | FastAPI + SSE |

## Project Structure

```
app/
├── config.py                  # All settings via pydantic-settings + .env
├── main.py                    # FastAPI entry point
├── pipeline.py                # End-to-end RAG pipeline orchestrator
├── models.py                  # Pydantic request/response schemas
├── preprocessing/
│   ├── pdf_processor.py       # PDF loading and text extraction
│   ├── chunker.py             # Parent/child + semantic chunking
│   ├── entity_extractor.py    # Medical entity extraction
│   └── rule_engine.py         # Domain-specific preprocessing rules
├── indexing/
│   └── indexer.py             # ChromaDB index management
├── retrieval/
│   ├── query_processor.py     # Query rewriting, HyDE, Multi-Query
│   ├── hybrid_retriever.py    # BM25 + vector dual-lane retrieval
│   ├── rrf.py                 # Dual-layer RRF fusion
│   └── reranker.py            # BGE cross-encoder reranking
├── generation/
│   ├── context_processor.py   # Context compression + citation building
│   └── generator.py           # DeepSeek streaming generation + Self-RAG
└── tenants/
    └── manager.py             # Multi-tenant isolation

scripts/
└── ingest.py                  # One-time PDF ingestion script
```

## Installation

```bash
pip install -r requirements.txt
```

Download local models:
- `BAAI/bge-m3` → embedding
- `BAAI/bge-reranker-v2-m3` → reranker

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

EMBEDDING_MODEL_PATH=/path/to/bge-m3
RERANKER_MODEL_PATH=/path/to/bge-reranker-v2-m3
CHROMA_DB_PATH=/path/to/chroma_db
BM25_INDEX_DIR=/path/to/bm25_index

PDF_FOLDER=folder
PDF_FILENAME=your_document.pdf
```

## Usage

**Step 1 — Ingest documents:**

```bash
python scripts/ingest.py
```

**Step 2 — Start the API server:**

```bash
uvicorn app.main:app --reload
```

**Step 3 — Query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of pneumonia?", "tenant_id": "default"}'
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieval_top_k` | 20 | Candidates per retrieval lane (BM25 & vector each) |
| `rrf_k` | 60 | RRF smoothing constant |
| `rerank_top_n` | 10 | Chunks passed to LLM after reranking |
| `multi_query_count` | 3 | Number of query variants |
| `parent_chunk_size` | 1000 | Parent chunk size in characters |
| `child_chunk_size` | 300 | Child chunk size for precise retrieval |
