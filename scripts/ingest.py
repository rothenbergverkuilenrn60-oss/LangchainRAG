"""
scripts/ingest.py — Command-line document ingestion script
Run this script directly to ingest a PDF document into the vector database.
Suitable for initial indexing or rebuilding the index without starting the FastAPI service.

Usage:
    python scripts/ingest.py                          # Use default config
    python scripts/ingest.py --tenant hospital_a      # Specify tenant
    python scripts/ingest.py --force                  # Force rebuild
    python scripts/ingest.py --pdf /path/to/file.pdf  # Specify PDF path
"""

import sys
import time
import pickle
import hashlib
import logging
import argparse
from pathlib import Path

# Add project root to Python path so `from app.xxx` imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.preprocessing.pdf_processor import PDFProcessor
from app.preprocessing.rule_engine import MedicalRuleEngine
from app.preprocessing.entity_extractor import MedicalEntityExtractor
from app.indexing.chunker import ParentChildChunker
from app.indexing.indexer import ChromaIndexer

# Configure logging (script-level, output to console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # ── Command-line argument parsing ─────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the Medical RAG vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py                              # Default settings
  python scripts/ingest.py --tenant hospital_a          # Custom tenant
  python scripts/ingest.py --force --tenant default     # Force rebuild
  python scripts/ingest.py --pdf "/path/to/manual.pdf"  # Custom file
        """,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help=f"PDF file path (default: {settings.pdf_folder}/{settings.pdf_filename})",
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=settings.default_tenant_id,
        help=f"Tenant ID (default: {settings.default_tenant_id})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild index (clears existing data first)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Vector write batch size (default: 128, reduce to 64 if VRAM is insufficient)",
    )
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="Generate summary index (requires DeepSeek API Key, takes longer for large docs)",
    )
    args = parser.parse_args()

    # Determine PDF path
    pdf_path = args.pdf or str(
        Path(settings.pdf_folder) / settings.pdf_filename
    )
    tenant_id = args.tenant

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Medical RAG — Document Ingestion Program")
    logger.info(f"  PDF file  : {pdf_path} ({pdf_file.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  Tenant ID : {tenant_id}")
    logger.info(f"  Vector DB : {settings.chroma_db_path}")
    logger.info(f"  Embedding : {settings.embedding_model_path}")
    logger.info(f"  Force rebuild: {args.force}")
    logger.info("=" * 60)

    total_start = time.time()

    # ── Step 1: Load embedding model ──────────────────────────────────────────
    logger.info("\n[1/5] Loading BGE-M3 embedding model...")
    t = time.time()
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(
        settings.embedding_model_path,
        device="cuda",   # Prefer GPU
    )
    logger.info(f"     Done ({time.time()-t:.1f}s)")

    # ── Step 2: Initialize indexer ────────────────────────────────────────────
    logger.info("\n[2/5] Initializing ChromaDB indexer...")
    t = time.time()
    indexer = ChromaIndexer(embedding_model)

    # If force rebuild, clear tenant data first
    if args.force:
        logger.info("     Clearing existing index...")
        indexer.reset_tenant(tenant_id)

    # Check for existing data
    stats = indexer.get_collection_stats(tenant_id)
    if stats.get("children", 0) > 0 and not args.force:
        logger.info(
            f"     Found existing {stats['children']} child chunk index; will resume from checkpoint (skipping already-indexed chunks). "
            "Use --force to fully rebuild."
        )
    logger.info(f"     Done ({time.time()-t:.1f}s)")

    # ── Step 3: PDF extraction and cleaning ───────────────────────────────────
    logger.info("\n[3/5] Parsing PDF document...")
    t = time.time()
    processor = PDFProcessor(pdf_path)
    pages = processor.extract()
    logger.info(f"     Extracted {len(pages)} pages ({time.time()-t:.1f}s)")

    # ── Step 4: Semantic chunking (parent-child) ──────────────────────────────
    logger.info("\n[4/5] Semantic chunking (parent + child chunks)...")
    logger.info(f"     Parent chunk size: {settings.parent_chunk_size} chars")
    logger.info(f"     Child chunk size: {settings.child_chunk_size} chars")
    t = time.time()

    # Chunk result cache: MD5 fingerprint based on PDF file stat + chunking parameters
    stat = pdf_file.stat()
    cache_key = hashlib.md5(
        f"{stat.st_size}_{stat.st_mtime}_{settings.parent_chunk_size}_"
        f"{settings.child_chunk_size}_{settings.semantic_breakpoint_threshold}".encode()
    ).hexdigest()
    cache_dir = Path(settings.bm25_index_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_cache_path = cache_dir / f"chunks_{cache_key}.pkl"

    if chunk_cache_path.exists() and not args.force:
        logger.info(f"     Chunk cache found, loading directly (skipping embedding computation)...")
        with open(chunk_cache_path, "rb") as f:
            parent_chunks, child_chunks = pickle.load(f)
        logger.info(
            f"     Parent: {len(parent_chunks)}, Child: {len(child_chunks)} "
            f"({time.time()-t:.1f}s, from cache)"
        )
    else:
        rule_engine = MedicalRuleEngine()
        entity_extractor = MedicalEntityExtractor(use_spacy=False)  # Disable spaCy in script for speed

        chunker = ParentChildChunker(
            embedding_model=embedding_model,
            rule_engine=rule_engine,
            entity_extractor=entity_extractor,
        )
        doc_source = pdf_file.name
        parent_chunks, child_chunks = chunker.create_chunks(
            pages=pages,
            tenant_id=tenant_id,
            doc_source=doc_source,
        )
        # Persist chunk results for reuse next time
        with open(chunk_cache_path, "wb") as f:
            pickle.dump((parent_chunks, child_chunks), f)
        logger.info(
            f"     Parent: {len(parent_chunks)}, Child: {len(child_chunks)} "
            f"({time.time()-t:.1f}s, cached to {chunk_cache_path.name})"
        )

    # ── Step 5: Write index ───────────────────────────────────────────────────
    logger.info("\n[5/5] Writing vector index (Chroma + BM25)...")
    logger.info(f"     Batch size: {args.batch_size}")
    t = time.time()
    total_indexed = indexer.index_chunks(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        tenant_id=tenant_id,
        batch_size=args.batch_size,
    )
    logger.info(f"     Written {total_indexed} chunks ({time.time()-t:.1f}s)")

    # ── Step 6: Generate summary index (optional) ─────────────────────────────
    if args.with_summary:
        logger.info("\n[6/6] Generating summary index (calling DeepSeek LLM)...")
        t = time.time()
        from langchain_deepseek import ChatDeepSeek
        from app.indexing.chunker import SummaryChunker
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
        indexer.index_summaries(summaries, tenant_id=tenant_id, batch_size=args.batch_size)
        logger.info(f"     Written {len(summaries)} summary chunks ({time.time()-t:.1f}s)")

    # ── Completion summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"  Total elapsed  : {total_elapsed/60:.1f} minutes")
    logger.info(f"  Parent chunks  : {len(parent_chunks)}")
    logger.info(f"  Child chunks   : {len(child_chunks)}")
    logger.info(f"  Total written  : {total_indexed}")
    final_stats = indexer.get_collection_stats(tenant_id)
    logger.info(f"  Collection stats: {final_stats}")
    logger.info("=" * 60)
    logger.info(f"\nYou can now start the API service:")
    logger.info("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
