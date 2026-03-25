"""
FastAPI dependency injection wiring for the Retrieval Service.

Chain: DB handle → Repository → Provider → Service → Route handler.
"""

from typing import Annotated
from fastapi import Depends
from pymongo.asynchronous.database import AsyncDatabase

from app.db.mongo import get_database
from app.providers.embedding.factory import EmbeddingProviderFactory
from app.providers.vectorstore.base import BaseVectorStoreProvider
from app.providers.vectorstore.factory import build_vectorstore_provider
from app.repositories.tenant_repo import TenantRepository
from app.repositories.repo_repo import RepoRepository
from app.services.filter_builder import FilterBuilder
from app.services.filter_extractor import FilterExtractor
from app.services.retrieval_service import RetrievalService

# ── Singletons — created once at startup ──────────────────────────────────────
# BaseVectorStoreProvider holds a sync MongoClient — expensive to create per request
_vector_store: BaseVectorStoreProvider | None = None
_filter_builder: FilterBuilder | None = None
_embedding_factory: EmbeddingProviderFactory | None = None
_filter_extractor: FilterExtractor | None = None


def init_singletons() -> None:
    """Called once at startup in main.py lifespan."""
    global _vector_store, _filter_builder
    _vector_store = build_vectorstore_provider()  # factory — easy to swap
    _filter_builder = FilterBuilder()
    _embedding_factory = EmbeddingProviderFactory()
    _filter_extractor = FilterExtractor()


def shutdown_singletons() -> None:
    """Called once at shutdown in main.py lifespan."""
    global _vector_store, _embedding_factory, _filter_extractor
    if _vector_store:
        _vector_store.close()
        _vector_store = None
    _embedding_factory = None
    _filter_extractor = None


def get_vector_store() -> BaseVectorStoreProvider:
    if _vector_store is None:
        raise RuntimeError("VectorStoreProvider not initialised.")
    return _vector_store


def get_filter_builder() -> FilterBuilder:
    if _filter_builder is None:
        raise RuntimeError("FilterBuilder not initialised.")
    return _filter_builder


def get_embedding_factory() -> EmbeddingProviderFactory:
    if _embedding_factory is None:
        raise RuntimeError("EmbeddingProviderFactory not initialised.")
    return _embedding_factory


def get_filter_extractor() -> FilterExtractor:
    if _filter_extractor is None:
        raise RuntimeError("FilterExtractor not initialised.")
    return _filter_extractor


# ── Per-request dependencies ──────────────────────────────────────────────────


def get_db() -> AsyncDatabase:
    return get_database()


def get_tenant_repo(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> TenantRepository:
    return TenantRepository(db)


def get_repo_repo(
    db: Annotated[AsyncDatabase, Depends(get_db)],
) -> RepoRepository:
    return RepoRepository(db)


def get_retrieval_service(
    tenant_repo: Annotated[TenantRepository, Depends(get_tenant_repo)],
    repo_repo: Annotated[RepoRepository, Depends(get_repo_repo)],
    vector_store: Annotated[BaseVectorStoreProvider, Depends(get_vector_store)],
    filter_builder: Annotated[FilterBuilder, Depends(get_filter_builder)],
    embedding_factory: Annotated[
        EmbeddingProviderFactory, Depends(get_embedding_factory)
    ],
    filter_extractor: Annotated[FilterExtractor, Depends(get_filter_extractor)],
) -> RetrievalService:
    return RetrievalService(
        tenant_repo=tenant_repo,
        repo_repo=repo_repo,
        vector_store=vector_store,
        filter_builder=filter_builder,
        embedding_factory=embedding_factory,
        filter_extractor=filter_extractor,
    )


# ── Type aliases ──────────────────────────────────────────────────────────────
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
