"""
REST API routes for the Retrieval Service.

Endpoints:
  POST /api/v1/{tenant_id}/search  → vector search, returns ranked chunks
  POST /api/v1/{tenant_id}/query   → RAG, returns LLM answer + source chunks
  GET  /api/v1/{tenant_id}/repos   → list active repos for tenant

All routes are tenant-scoped via path param.
Ocelot handles authentication — tenant_id is trusted from the path.
"""

import structlog
from fastapi import APIRouter, HTTPException, status
from typing import Annotated
from fastapi import Depends

from app.api.dependencies import RetrievalServiceDep
from app.core.api_config import resolve_api_config
from app.core.exceptions import (
    TenantNotFoundError,
    TenantInactiveError,
    NoActiveReposError,
    EmbeddingError,
    VectorSearchError,
    LLMError,
)
from app.repositories.tenant_repo import TenantRepository
from app.repositories.repo_repo import RepoRepository
from app.schemas.requests import SearchRequest, QueryRequest
from app.schemas.responses import (
    SearchResponse,
    QueryResponse,
    ReposResponse,
    RepoSummary,
    ChunkResponse,
)
from app.services.llm_service import LLMService

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


def _handle_domain_error(e: Exception, tenant_id: str) -> None:
    """Map domain exceptions to HTTP responses."""
    if isinstance(e, TenantNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if isinstance(e, TenantInactiveError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    if isinstance(e, NoActiveReposError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if isinstance(e, (EmbeddingError, VectorSearchError, LLMError)):
        log.error("retrieval.service_error", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream service error. Please try again.",
        )
    raise e


# ── POST /api/v1/{tenant_id}/search ──────────────────────────────────────────


@router.post(
    "/{tenant_id}/search",
    response_model=SearchResponse,
    summary="Vector search — returns ranked chunks",
    description=(
        "Embeds the question and performs $vectorSearch across all active repos "
        "for the tenant. Returns ranked document chunks with similarity scores. "
        "No LLM call is made."
    ),
)
async def search(
    tenant_id: str,
    body: SearchRequest,
    svc: RetrievalServiceDep,
) -> SearchResponse:
    try:
        # Explicit tenant validation — raises TenantNotFoundError / TenantInactiveError
        await svc._tenant_repo.get_by_id(tenant_id)

        chunks, skipped_filters = await svc.search(
            question=body.question,
            tenant_id=tenant_id,
            filters=body.filters,
            top_k=body.top_k,
            repo_ids=body.repo_ids,
        )
        repos_searched = len({c.get("repo_id") for c in chunks})

        return SearchResponse(
            question=body.question,
            chunks=[ChunkResponse.from_dict(c) for c in chunks],
            total=len(chunks),
            repos_searched=repos_searched,
            skipped_filters=skipped_filters,
        )
    except Exception as e:
        _handle_domain_error(e, tenant_id)


# ── POST /api/v1/{tenant_id}/query ───────────────────────────────────────────


@router.post(
    "/{tenant_id}/query",
    response_model=QueryResponse,
    summary="RAG query — returns LLM answer + source chunks",
    description=(
        "Retrieves relevant chunks via vector search, then passes them as context "
        "to the tenant's configured LLM to generate a grounded answer."
    ),
)
async def query(
    tenant_id: str,
    body: QueryRequest,
    svc: RetrievalServiceDep,
) -> QueryResponse:
    try:
        # Step 1 — retrieve chunks
        chunks, _ = await svc.search(
            question=body.question,
            tenant_id=tenant_id,
            filters=body.filters,
            top_k=body.top_k,
            repo_ids=body.repo_ids,
        )

        # Step 2 — resolve tenant api_config for LLM call
        tenant = await svc._tenant_repo.get_by_id(tenant_id)
        api_cfg = await resolve_api_config(
            tenant_api_config=tenant.get("api_config"),
            tenant_ingestion_defaults=tenant.get("ingestion_defaults", {}),
        )

        # Step 3 — generate answer
        llm_svc = LLMService()
        answer = await llm_svc.generate_answer(
            question=body.question,
            chunks=chunks,
            api_cfg=api_cfg,
        )

        return QueryResponse(
            question=body.question,
            answer=answer,
            chunks=[ChunkResponse.from_dict(c) for c in chunks],
            total_chunks=len(chunks),
        )
    except Exception as e:
        _handle_domain_error(e, tenant_id)


# ── GET /api/v1/{tenant_id}/repos ────────────────────────────────────────────


@router.get(
    "/{tenant_id}/repos",
    response_model=ReposResponse,
    summary="List active repos for tenant",
    description="Returns all active repositories for the tenant with their filterable fields.",
)
async def list_repos(
    tenant_id: str,
    svc: RetrievalServiceDep,
) -> ReposResponse:
    try:
        # Verify tenant exists first
        await svc._tenant_repo.get_by_id(tenant_id)

        repos = await svc._repo_repo.list_for_tenant(tenant_id)
        return ReposResponse(
            tenant_id=tenant_id,
            repos=[
                RepoSummary(
                    repo_id=str(r["_id"]),
                    name=r.get("name"),
                    source_type=r.get("source_type", "unknown"),
                    filterable_fields=r.get("retrieval_config", {}).get(
                        "filterable_fields", []
                    ),
                    extractable_fields=r.get("retrieval_config", {}).get(
                        "extractable_fields", []
                    ),
                )
                for r in repos
            ],
            total=len(repos),
        )
    except Exception as e:
        _handle_domain_error(e, tenant_id)
