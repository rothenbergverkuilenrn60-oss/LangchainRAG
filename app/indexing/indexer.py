"""
indexer.py — Multi-collection Chroma indexer
Responsible for writing DocumentChunk lists to ChromaDB and maintaining the BM25 inverted index.

Collection design (independent per tenant):
  - {prefix}_{tenant}_children : child chunk vector collection (main retrieval target)
  - {prefix}_{tenant}_parents  : parent chunk full-text storage (queried by chunk_id)
  - {prefix}_{tenant}_summaries: section summary vector collection (macro-level retrieval)

BM25 index:
  - Each tenant maintains an independent BM25 index (based on child chunk text)
  - Serialized to disk (pickle), supports persistent reuse
"""

import os
import pickle
import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
import jieba                      # Chinese word segmentation, tokenizer for BM25

from app.models import DocumentChunk
from app.config import settings

logger = logging.getLogger(__name__)

# Chroma metadata value type restriction: only supports str/int/float/bool
# Complex types (list/dict) must be serialized to JSON strings
import json


def _serialize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts unsupported types (list/dict) in metadata to JSON strings,
    making them compliant with Chroma's metadata value type constraints.
    """
    serialized = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        elif isinstance(v, (list, dict)):
            serialized[k] = json.dumps(v, ensure_ascii=False)   # ensure_ascii=False: preserves Chinese characters
        elif v is None:
            serialized[k] = ""
        else:
            serialized[k] = str(v)
    return serialized


def _chunk_to_metadata(chunk: DocumentChunk) -> Dict[str, Any]:
    """
    Extracts DocumentChunk fields into a Chroma metadata dictionary.
    Used for subsequent metadata-filtered retrieval (e.g., filter by section, page number).
    """
    return _serialize_metadata({
        "chunk_id": chunk.chunk_id,
        "parent_chunk_id": chunk.parent_chunk_id or "",
        "page_numbers": chunk.page_numbers,         # Will be serialized to JSON str
        "section_title": chunk.section_title,
        "chapter": chunk.chapter,
        "tenant_id": chunk.tenant_id,
        "doc_source": chunk.doc_source,
        "entities": chunk.entities,                 # Will be serialized to JSON str
        "has_table": chunk.metadata.get("has_table", False),
        "has_warning": chunk.metadata.get("has_warning", False),
        "importance_boost": chunk.metadata.get("importance_boost", 1.0),
        "is_low_quality": chunk.metadata.get("is_low_quality", False),
        "chunk_type": chunk.metadata.get("chunk_type", "child"),
        "char_count": chunk.metadata.get("char_count", len(chunk.text)),
        # Figure chunk specific fields
        "is_figure": chunk.metadata.get("is_figure", False),
        "image_path": chunk.metadata.get("image_path", ""),
        "figure_caption": chunk.metadata.get("figure_caption", ""),
        "figure_number": chunk.metadata.get("figure_number", ""),
    })


class ChromaIndexer:
    """
    Multi-collection ChromaDB indexer + BM25 inverted index manager.

    Each tenant has an independent namespace (collection) for data isolation.
    Embedding vectors are computed by an external embedding_model (not Chroma's built-in embedding function),
    to reuse the already-loaded BGE-M3 model instance and save VRAM.

    Usage:
        indexer = ChromaIndexer(embedding_model)
        indexer.index_chunks(parent_chunks, child_chunks, tenant_id="default")
    """

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: SentenceTransformer instance (BGE-M3)
        """
        self.emb_model = embedding_model

        # Initialize ChromaDB persistent client
        # PersistentClient automatically writes data to disk
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Disable anonymous telemetry reporting
                allow_reset=True,            # Allow reset (for development/testing)
            ),
        )
        logger.info(f"ChromaDB client initialized: {settings.chroma_db_path}")

        # BM25 index storage directory
        self._bm25_dir = Path(settings.bm25_index_dir)
        self._bm25_dir.mkdir(parents=True, exist_ok=True)

        # BM25 index cache: {tenant_id: (BM25Okapi, List[str chunk_ids], List[str texts])}
        self._bm25_cache: Dict[str, Tuple[BM25Okapi, List[str], List[str]]] = {}

        # Parent chunk text cache: {tenant_id: {chunk_id: {"text": ..., "metadata": ...}}}
        # Parent chunks are only queried by ID, no vector index needed;
        # stored as pickle to save significant embedding computation
        self._parent_cache: Dict[str, Dict[str, Dict]] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Collection name generation
    # ─────────────────────────────────────────────────────────────────────

    def _col_children(self, tenant_id: str) -> str:
        """Child chunk collection name"""
        return f"{settings.collection_prefix}_{tenant_id}_children"

    def _col_parents(self, tenant_id: str) -> str:
        """Parent chunk collection name"""
        return f"{settings.collection_prefix}_{tenant_id}_parents"

    def _col_summaries(self, tenant_id: str) -> str:
        """Summary collection name (for macro-level retrieval)"""
        return f"{settings.collection_prefix}_{tenant_id}_summaries"

    def _parent_store_path(self, tenant_id: str) -> Path:
        """Parent chunk pickle file path"""
        return self._bm25_dir / f"{tenant_id}_parents.pkl"

    def _load_parent_store(self, tenant_id: str) -> Dict[str, Dict]:
        """Loads parent chunk store (in-memory cache first, then disk)"""
        if tenant_id in self._parent_cache:
            return self._parent_cache[tenant_id]
        path = self._parent_store_path(tenant_id)
        if path.exists():
            with open(path, "rb") as f:
                store = pickle.load(f)
            self._parent_cache[tenant_id] = store
            return store
        return {}

    def _save_parent_store(self, store: Dict[str, Dict], tenant_id: str):
        """Persists parent chunk store to disk"""
        path = self._parent_store_path(tenant_id)
        with open(path, "wb") as f:
            pickle.dump(store, f)
        self._parent_cache[tenant_id] = store

    # ─────────────────────────────────────────────────────────────────────
    # Core indexing methods
    # ─────────────────────────────────────────────────────────────────────

    def index_chunks(
        self,
        parent_chunks: List[DocumentChunk],
        child_chunks: List[DocumentChunk],
        tenant_id: str,
        batch_size: int = 64,
    ) -> int:
        """
        Writes parent and child chunks to Chroma, and builds/updates the BM25 index.

        Args:
            parent_chunks: Parent chunk list (full-text storage)
            child_chunks: Child chunk list (main vector retrieval target)
            tenant_id: Tenant ID
            batch_size: Chunks per write batch (controls memory usage)

        Returns:
            Total chunks written
        """
        # Get or create child chunk collection
        children_col = self.client.get_or_create_collection(
            name=self._col_children(tenant_id),
            # Use cosine distance (BGE-M3 normalized cosine distance = 1 - inner product)
            metadata={"hnsw:space": "cosine"},
        )

        total_indexed = 0

        # ── Store parent chunks (pickle, no embedding needed) ──────────────────
        # Parent chunks are only queried by chunk_id, never vector-searched;
        # skip embedding to save significant time
        logger.info(f"Storing {len(parent_chunks)} parent chunks (pickle, skipping embedding)...")
        parent_store = {
            c.chunk_id: {"text": c.text, "metadata": _chunk_to_metadata(c)}
            for c in parent_chunks
        }
        self._save_parent_store(parent_store, tenant_id)
        total_indexed += len(parent_chunks)
        logger.info(f"  Parent chunk storage complete")

        # ── Write child chunks (main retrieval target) ─────────────────────────
        logger.info(f"Writing {len(child_chunks)} child chunks...")
        # Filter out low-quality chunks (flagged by rule engine)
        valid_children = [
            c for c in child_chunks
            if not c.metadata.get("is_low_quality", False)
        ]
        logger.info(f"After filtering low-quality chunks: {len(valid_children)} child chunks")
        total_indexed += self._batch_upsert(
            collection=children_col,
            chunks=valid_children,
            batch_size=batch_size,
        )

        # ── Build/update BM25 index ───────────────────────────────────────────
        logger.info("Building BM25 inverted index...")
        self._build_bm25_index(valid_children, tenant_id)

        logger.info(f"Indexing complete, {total_indexed} chunks written")
        return total_indexed

    def index_summaries(
        self,
        summaries: List[Dict[str, Any]],
        tenant_id: str,
        batch_size: int = 32,
    ):
        """
        Writes section summaries to the separate summaries collection.
        summaries format: [{"chunk_id": ..., "text": ..., "metadata": ...}]
        """
        if not summaries:
            return
        col = self.client.get_or_create_collection(
            name=self._col_summaries(tenant_id),
            metadata={"hnsw:space": "cosine"},
        )
        ids = [s["chunk_id"] for s in summaries]
        texts = [s["text"] for s in summaries]
        metadatas = [_serialize_metadata(s.get("metadata", {})) for s in summaries]

        # Batch compute embeddings
        embeddings = self.emb_model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()  # .tolist(): converts numpy/torch tensor to Python list

        # Write in batches
        for i in range(0, len(ids), batch_size):
            col.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        logger.info(f"Written {len(summaries)} summary chunks")

    # ─────────────────────────────────────────────────────────────────────
    # Query interface (called by Retriever)
    # ─────────────────────────────────────────────────────────────────────

    def vector_search(
        self,
        query_text: str,
        tenant_id: str,
        top_k: int = None,
        where: Optional[Dict] = None,
        collection_type: str = "children",
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity retrieval.

        Args:
            query_text: Query text (encoded to vector)
            tenant_id: Tenant ID
            top_k: Number of results to return
            where: Chroma metadata filter conditions (e.g., {"chapter": "Cardiology"})
            collection_type: "children" | "parents" | "summaries"

        Returns:
            List[Dict], each dict contains text, metadata, distance
        """
        top_k = top_k or settings.retrieval_top_k

        # Compute query vector
        query_emb = self.emb_model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Select collection
        col_name = {
            "children": self._col_children(tenant_id),
            "parents": self._col_parents(tenant_id),
            "summaries": self._col_summaries(tenant_id),
        }[collection_type]

        try:
            col = self.client.get_collection(col_name)
        except Exception:
            logger.warning(f"Collection {col_name} not found, please run ingest first")
            return []

        # Execute vector retrieval
        kwargs = {
            "query_embeddings": query_emb,
            "n_results": min(top_k, col.count()),  # Don't exceed collection size
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        # Format return data
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                # Convert distance to similarity score (cosine distance ∈ [0,2], score ∈ [-1,1])
                "score": 1.0 - dist,
            })
        return output

    def get_parent_by_id(
        self,
        parent_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Gets parent chunk full text by chunk_id (for context expansion).
        Reads from pickle store, no vector retrieval needed.
        """
        try:
            store = self._load_parent_store(tenant_id)
            return store.get(parent_id)
        except Exception as e:
            logger.warning(f"Failed to get parent chunk {parent_id}: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # BM25 related
    # ─────────────────────────────────────────────────────────────────────

    def bm25_search(
        self,
        query_text: str,
        tenant_id: str,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25 keyword retrieval.

        Args:
            query_text: Query text
            tenant_id: Tenant ID
            top_k: Number of results to return

        Returns:
            List[Dict], containing text, metadata, score
        """
        top_k = top_k or settings.retrieval_top_k
        bm25_data = self._load_bm25_index(tenant_id)
        if bm25_data is None:
            logger.warning(f"BM25 index for tenant {tenant_id} not found")
            return []

        bm25, chunk_ids, texts, metadatas_list = bm25_data

        # Tokenize query
        query_tokens = list(jieba.cut(query_text))

        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)  # BM25 score list, one element per child chunk

        # Take Top-K
        '''
        scores = [0.1, 2.3, 0.5, 1.7]
        range(len(scores)) = [0, 1, 2, 3]
        i=0 → key= scores[0] = 0.1
        i=1 → key= scores[1] = 2.3
        '''
        top_indices = sorted(
            range(len(scores)),         # len(scores): length of BM25 score list
            key=lambda i: scores[i],    # scores[i]: BM25 score for the i-th child chunk
            reverse=True,               # Sort by BM25 score descending
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:        # Skip chunks with zero or negative score
                continue
            results.append({
                "text": texts[idx],
                "metadata": metadatas_list[idx],
                "score": float(scores[idx]),
                # chunk_id field added here (not in vector_search results) since
                # vector search IDs are already in metadata; BM25 adds it explicitly for RRF alignment
                "chunk_id": chunk_ids[idx],
            })
        return results

    def _build_bm25_index(self, chunks: List[DocumentChunk], tenant_id: str):
        """
        Builds a BM25 index from child chunk list and persists to disk.
        If child chunk content hasn't changed (fingerprint match), skips rebuild.

        tokenizer uses jieba (Chinese word segmentation) to ensure Chinese keywords
        can be recognized by BM25.
        """
        chunk_ids = [c.chunk_id for c in chunks]
        index_path = self._bm25_dir / f"{tenant_id}_bm25.pkl"
        fp_path = self._bm25_dir / f"{tenant_id}_bm25.fp"

        # Compute MD5 fingerprint of current child chunk set
        fingerprint = hashlib.md5("".join(chunk_ids).encode()).hexdigest()

        # Skip rebuild if fingerprint matches
        if fp_path.exists() and index_path.exists():
            if fp_path.read_text().strip() == fingerprint:
                logger.info("BM25 index content unchanged (fingerprint match), skipping rebuild")
                if tenant_id not in self._bm25_cache:
                    self._load_bm25_index(tenant_id)
                return

        texts = [c.text for c in chunks]
        metadatas_list = [_chunk_to_metadata(c) for c in chunks]

        # jieba tokenization (cut_all=False uses precise mode)
        tokenized = [list(jieba.cut(t)) for t in texts]

        # Build BM25Okapi (classic BM25 variant, parameters k1=1.5, b=0.75)
        bm25 = BM25Okapi(tokenized)

        # Persist index and fingerprint
        with open(index_path, "wb") as f:
            pickle.dump((bm25, chunk_ids, texts, metadatas_list), f)
        fp_path.write_text(fingerprint)

        # Update in-memory cache
        self._bm25_cache[tenant_id] = (bm25, chunk_ids, texts, metadatas_list)
        logger.info(f"BM25 index saved: {index_path} ({len(texts)} entries)")

    def _load_bm25_index(self, tenant_id: str):
        """Loads BM25 index from cache or disk"""
        if tenant_id in self._bm25_cache:
            return self._bm25_cache[tenant_id]

        index_path = self._bm25_dir / f"{tenant_id}_bm25.pkl"
        if not index_path.exists():
            return None

        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25_cache[tenant_id] = data
        return data

    # ─────────────────────────────────────────────────────────────────────
    # Private utility methods
    # ─────────────────────────────────────────────────────────────────────

    def _batch_upsert(
        self,
        collection,
        chunks: List[DocumentChunk],
        batch_size: int,
    ) -> int:
        """
        Writes chunks to Chroma collection in batches.
        Uses upsert (update if exists, insert if not) to ensure idempotency.
        Supports checkpoint resumption: existing chunk_ids are skipped without recomputing embeddings.
        """
        # Get all existing IDs in the collection for checkpoint resumption
        try:
            existing_ids = set(collection.get(include=[])["ids"])
        except Exception:
            existing_ids = set()

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            logger.info(f"  Checkpoint resumption: skipping {skipped} already-indexed, {len(new_chunks)} to write")

        total = skipped
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [_chunk_to_metadata(c) for c in batch]

            # Batch compute vectors
            embeddings = self.emb_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2 normalization, cosine distance more accurate
            ).tolist()

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info(f"  Written {total}/{len(chunks)}")

        return total

    def get_collection_stats(self, tenant_id: str) -> Dict[str, int]:
        """Returns document count statistics for each collection"""
        stats = {}
        for col_type, col_name in [
            ("children", self._col_children(tenant_id)),
            ("summaries", self._col_summaries(tenant_id)),
        ]:
            try:
                col = self.client.get_collection(col_name)
                stats[col_type] = col.count()
            except Exception:
                stats[col_type] = 0
        # Count parent chunks from pickle
        store = self._load_parent_store(tenant_id)
        stats["parents"] = len(store)
        return stats

    def reset_tenant(self, tenant_id: str):
        """Deletes all collections and BM25 index for a specified tenant (dangerous operation!)"""
        for col_name in [
            self._col_children(tenant_id),
            self._col_summaries(tenant_id),
        ]:
            try:
                self.client.delete_collection(col_name)
                logger.info(f"Collection deleted: {col_name}")
            except Exception:
                pass
        # Delete BM25 index file
        idx_path = self._bm25_dir / f"{tenant_id}_bm25.pkl"
        if idx_path.exists():
            idx_path.unlink()
        self._bm25_cache.pop(tenant_id, None)
        # Delete parent chunk pickle file
        parent_path = self._parent_store_path(tenant_id)
        if parent_path.exists():
            parent_path.unlink()
        self._parent_cache.pop(tenant_id, None)
