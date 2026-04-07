"""
tenants/manager.py — Multi-tenant manager
Implements data isolation based on tenant ID, each tenant having independent:
  - Chroma vector collections (different namespaces)
  - BM25 indexes (independent disk files)
  - Configuration parameters (e.g., custom filter rules, top_k, etc.)

Design pattern: Singleton TenantManager, globally manages lifecycle of all tenants.

Multi-tenant use cases:
  - tenant "hospital_a": Hospital A's private document library
  - tenant "hospital_b": Hospital B's private document library
  - tenant "default": Shared Merck Manual

Data isolation guarantee: queries from different tenants are completely independent and invisible to each other.
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tenant configuration model
# ─────────────────────────────────────────────────────────────────────────────

class TenantConfig(BaseModel):
    """
    Configuration info for a single tenant.
    Can override global defaults (e.g., custom top_k, temperature, etc.).
    """
    tenant_id: str
    description: str = ""                   # Tenant description (e.g., "Hospital A Internal Medicine")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True                  # Whether enabled; False can reject requests from this tenant

    # Tenant-level parameters that can override global config
    retrieval_top_k: Optional[int] = None   # None = use global config
    rerank_top_n: Optional[int] = None
    llm_temperature: Optional[float] = None

    # Tenant-specific metadata filters (always in effect, no need to pass per request)
    default_filters: Dict[str, Any] = {}

    # Statistics
    total_queries: int = 0                  # Total query count
    total_docs_indexed: int = 0             # Total document chunks indexed
    last_query_at: Optional[str] = None     # Last query time
    last_ingest_at: Optional[str] = None    # Last ingestion time


# ─────────────────────────────────────────────────────────────────────────────
# Multi-tenant manager (singleton)
# ─────────────────────────────────────────────────────────────────────────────

class TenantManager:
    """
    Multi-tenant manager: handles tenant registration, querying, statistics, and lifecycle.

    Usage:
        manager = TenantManager()  # Use singleton
        config = manager.get_or_create("hospital_a")
        manager.record_query("hospital_a")
    """

    _instance = None
    _lock = threading.Lock()  # Thread-safe singleton lock

    def __new__(cls):
        """Double-checked locking singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # Prevent duplicate initialization
        self._initialized = True

        # Tenant registry: {tenant_id: TenantConfig}
        self._tenants: Dict[str, TenantConfig] = {}
        self._registry_lock = threading.RLock()  # Reentrant lock protecting _tenants

        # Pre-register the default tenant
        self._register_default_tenant()
        logger.info("TenantManager initialized")

    def _register_default_tenant(self):
        """Registers the built-in default tenant (shared document library)"""
        default_id = settings.default_tenant_id
        if default_id not in self._tenants:
            self._tenants[default_id] = TenantConfig(
                tenant_id=default_id,
                description="Default shared tenant (Merck Manual)",
            )

    # ─────────────────────────────────────────────────────────────────────
    # Tenant lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def get_or_create(
        self,
        tenant_id: str,
        description: str = "",
        **config_kwargs,
    ) -> TenantConfig:
        """
        Gets an existing tenant config, or automatically creates a new tenant.
        Thread-safe.

        Args:
            tenant_id: Tenant ID (alphanumeric and underscores)
            description: Tenant description
            **config_kwargs: Other TenantConfig fields

        Returns:
            TenantConfig
        """
        self._validate_tenant_id(tenant_id)

        with self._registry_lock:
            if tenant_id not in self._tenants:
                config = TenantConfig(
                    tenant_id=tenant_id,
                    description=description,
                    **config_kwargs,
                )
                self._tenants[tenant_id] = config
                logger.info(f"New tenant registered: {tenant_id}")
            return self._tenants[tenant_id]

    def get(self, tenant_id: str) -> Optional[TenantConfig]:
        """Gets tenant config; returns None if not found"""
        return self._tenants.get(tenant_id)

    def exists(self, tenant_id: str) -> bool:
        """Checks whether a tenant exists"""
        return tenant_id in self._tenants

    def list_tenants(self) -> List[TenantConfig]:
        """Lists all active tenants"""
        with self._registry_lock:
            return [
                config for config in self._tenants.values()
                if config.is_active
            ]

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivates a tenant (soft delete, does not delete data)"""
        with self._registry_lock:
            if tenant_id not in self._tenants:
                return False
            self._tenants[tenant_id].is_active = False
            logger.info(f"Tenant deactivated: {tenant_id}")
            return True

    # ─────────────────────────────────────────────────────────────────────
    # Statistics recording
    # ─────────────────────────────────────────────────────────────────────

    def record_query(self, tenant_id: str):
        """Records a query (for statistics and monitoring)"""
        with self._registry_lock:
            if tenant_id in self._tenants:
                config = self._tenants[tenant_id]
                config.total_queries += 1
                config.last_query_at = datetime.now().isoformat()

    def record_ingest(self, tenant_id: str, docs_count: int):
        """Records a document ingestion"""
        with self._registry_lock:
            if tenant_id in self._tenants:
                config = self._tenants[tenant_id]
                config.total_docs_indexed += docs_count
                config.last_ingest_at = datetime.now().isoformat()

    # ─────────────────────────────────────────────────────────────────────
    # Config merging
    # ─────────────────────────────────────────────────────────────────────

    def get_effective_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Gets the effective configuration for a tenant (tenant-level overrides global config).
        Used to get the correct hyperparameters during retrieval and generation.

        Returns:
            Merged configuration dictionary
        """
        global_cfg = {
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "llm_temperature": settings.llm_temperature,
            "rrf_k": settings.rrf_k,
        }

        tenant_cfg = self.get(tenant_id)
        if not tenant_cfg:
            return global_cfg

        # Tenant config overrides global (None means use global default)
        if tenant_cfg.retrieval_top_k is not None:
            global_cfg["retrieval_top_k"] = tenant_cfg.retrieval_top_k
        if tenant_cfg.rerank_top_n is not None:
            global_cfg["rerank_top_n"] = tenant_cfg.rerank_top_n
        if tenant_cfg.llm_temperature is not None:
            global_cfg["llm_temperature"] = tenant_cfg.llm_temperature

        global_cfg["default_filters"] = tenant_cfg.default_filters
        return global_cfg

    # ─────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_tenant_id(tenant_id: str):
        """
        Validates tenant ID format: only letters, numbers, underscores, and hyphens allowed.
        Prevents path injection attacks.
        """
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', tenant_id):
            raise ValueError(
                f"Invalid tenant ID format: '{tenant_id}'. "
                "Only letters, numbers, underscores, and hyphens are allowed, length 1-64."
            )


# Global singleton instance (other modules import this instance to use)
tenant_manager = TenantManager()
