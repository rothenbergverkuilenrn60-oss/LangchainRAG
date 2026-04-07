"""
config.py — Global configuration center
Uses pydantic-settings to automatically read config from environment variables / .env file.
All paths, model names, and hyperparameters are managed here.
Other modules only need to import Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # ── .env file path (project root .env) ──────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # Read variables from .env file first
        env_file_encoding="utf-8",
        extra="ignore",           # Ignore extra keys in .env without raising errors
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    deepseek_api_key: str = ""                          # DeepSeek API Key
    deepseek_base_url: str = "https://api.deepseek.com" # DeepSeek base URL
    llm_model_name: str = "deepseek-chat"               # LLM model name
    llm_temperature: float = 0.1                        # Generation temperature (lower = more deterministic)
    llm_max_tokens: int = 2048                          # Max tokens per generation

    # ── Local model paths ────────────────────────────────────────────────────
    embedding_model_path: str = "/mnt/f/my_models/embedding_models/bge-m3/BAAI/bge-m3"
    reranker_model_path: str = "/mnt/f/my_models/embedding_models/bge-m3-rerank/BAAI/bge-reranker-v2-m3"

    # ── Vector database ──────────────────────────────────────────────────────
    chroma_db_path: str = "/mnt/f/my_models/vector_store/chroma_db"
    # Each tenant uses {collection_prefix}_{tenant_id} as its collection name
    collection_prefix: str = "merck_rag"

    # ── Document paths ───────────────────────────────────────────────────────
    pdf_folder: str = "floder"                          # PDF directory (relative path)
    pdf_filename: str = "默克诊疗手册 第20版 上卷 王卫平译  2021.pdf"

    # ── Chunking hyperparameters ─────────────────────────────────────────────
    parent_chunk_size: int = 1000   # Parent chunk size (chars), provides full context
    parent_chunk_overlap: int = 100  # Parent chunk overlap
    child_chunk_size: int = 300     # Child chunk size (chars), for precise retrieval
    child_chunk_overlap: int = 30   # Child chunk overlap
    # Semantic chunking threshold: split when cosine distance between adjacent sentences exceeds this
    semantic_breakpoint_threshold: float = 0.45

    # ── Retrieval hyperparameters ────────────────────────────────────────────
    retrieval_top_k: int = 20       # Top-K recall per retrieval lane (BM25 & vector each)
    rrf_k: int = 60                 # Smoothing constant in RRF formula (classic value 60)
    rerank_top_n: int = 10          # Final reranker output count (chunks fed to LLM)
    multi_query_count: int = 3      # Number of query variants for Multi-Query

    # ── Multi-tenancy ────────────────────────────────────────────────────────
    default_tenant_id: str = "default"   # Default tenant when none is specified

    # ── BM25 persistence directory ───────────────────────────────────────────
    bm25_index_dir: str = "/mnt/f/my_models/vector_store/bm25_index"

    # ── Image storage directory (same level as bm25_index) ──────────────────
    pic_dir: str = "/mnt/f/my_models/vector_store/pic"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"


# Singleton: global Settings instance, other modules use `from app.config import settings`
settings = Settings()
