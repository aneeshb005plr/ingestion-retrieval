"""
REST API routes for the Retrieval Service.

Endpoints:
  POST /api/v1/{tenant_id}/search  → vector search, returns ranked chunks
  POST /api/v1/{tenant_id}/query   → RAG, returns LLM answer + cited chunks only
  GET  /api/v1/{tenant_id}/repos   → list active repos for tenant

All routes are tenant-scoped via path param.
Ocelot handles authentication — tenant_id is trusted from the path.

── Query endpoint change (R7) ───────────────────────────────────────────────
Before R7:
  - LLM received all top_k chunks and returned a plain text answer
  - All chunks were returned in the response regardless of LLM usage

After R7:
  - LLM receives numbered chunks and returns structured JSON
  - Response includes ONLY chunks the LLM cited (used_chunk_indices)
  - answer_available=False → answer=None, chunks=[]
  - parse_failed fallback → answer=raw text, chunks=[] (safe, logged)
─────────────────────────────────────────────────────────────────────────────
"""

import structlog
from fastapi import APIRouter, HTTPException, status

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
    summary="RAG query — returns LLM answer + cited chunks only",
    description=(
        "Retrieves relevant chunks via vector search, then passes them as numbered "
        "context to the tenant's LLM. Returns a structured answer with only the "
        "chunks the LLM actually cited. When the answer is not available in the "
        "documents, answer_available=False and chunks=[]."
    ),
)
async def query(
    tenant_id: str,
    body: QueryRequest,
    svc: RetrievalServiceDep,
) -> QueryResponse:
    try:
        bound_log = log.bind(tenant_id=tenant_id)

        # ── Step 1: Retrieve + rerank chunks ─────────────────────────────────
        chunks, skipped_filters = await svc.search(
            question=body.question,
            tenant_id=tenant_id,
            filters=body.filters,
            top_k=body.top_k,
            repo_ids=body.repo_ids,
        )

        # ── Step 2: Resolve tenant api_config for LLM ────────────────────────
        tenant = await svc._tenant_repo.get_by_id(tenant_id)
        api_cfg = await resolve_api_config(
            tenant_api_config=tenant.get("api_config"),
            tenant_ingestion_defaults=tenant.get("ingestion_defaults", {}),
        )

        # ── Step 3: Generate structured answer ───────────────────────────────
        # LLMService returns a typed LLMResult with:
        #   answer        — text or None
        #   answer_found  — bool
        #   used_indices  — 0-based indices into `chunks`
        #   parse_failed  — True if JSON from LLM was malformed (fallback used)
        llm_svc = LLMService()
        result = await llm_svc.generate_answer(
            question=body.question,
            chunks=chunks,
            api_cfg=api_cfg,
        )

        # ── Step 4: Build cited chunk list ────────────────────────────────────
        # Filter the full chunk list to only those the LLM cited.
        # When answer_found=False or parse_failed → cited = [] (no attribution).
        # Index safety: used_indices are already clamped in LLMService._parse_llm_response.
        if result.answer_found and not result.parse_failed:
            cited_chunks = [chunks[i] for i in result.used_indices]
        else:
            cited_chunks = []

        bound_log.info(
            "query.completed",
            answer_available=result.answer_found,
            chunks_retrieved=len(chunks),
            chunks_cited=len(cited_chunks),
            parse_failed=result.parse_failed,
        )

        # ── Step 5: Build response ────────────────────────────────────────────
        return QueryResponse(
            question=body.question,
            answer=result.answer,
            answer_available=result.answer_found,
            chunks=[ChunkResponse.from_dict(c) for c in cited_chunks],
            cited_chunks=len(cited_chunks),
            skipped_filters=skipped_filters,
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