"""
MCP Tools — the 3 tools exposed to AI agents.

Design principle: GENERAL PURPOSE — not tied to any specific tenant.

Each tenant has its own filterable_fields and extractable_fields.
Tools use a generic filters dict so any tenant can filter by
their own field names without code changes.

Agent workflow:
  1. list_repos()
     → discovers available knowledge bases + filterable_fields
     → learns what filter keys to use for this tenant

  2. search_documents(question, filters={...})
     OR query_knowledge_base(question, filters={...})
     → passes filters using field names from list_repos()

Examples across tenants:
  DocAssist:   filters={"application": "Smart Pricing Tool", "domain": "XLOS"}
  SmartQuery:  filters={"table_name": "customers", "schema_name": "sales"}
  HR:          filters={"department": "Finance", "policy_type": "leave"}
  Finance:     filters={"business_unit": "Tax", "quarter": "Q1"}

Filter semantics:
  Values within a field → ORed  (match ANY value)
  Fields across dims    → ANDed (match ALL dimensions)

  {"application": ["SPT", "Flex Forecast"], "domain": "XLOS"}
  → (app=SPT OR app=Flex Forecast) AND domain=XLOS

Tenant scoping:
  tenant_id baked in via closure at mount time.
  Tool calls never need to pass tenant_id.
  One MCP mount = one tenant = one data boundary.
"""

from typing import Any
import structlog

from app.schemas.search import SearchFilters
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.core.api_config import resolve_api_config

log = structlog.get_logger(__name__)


def make_tools(tenant_id: str, retrieval_service: RetrievalService):
    """
    Factory that returns the 3 tool functions pre-scoped to a tenant.

    Returns:
        Tuple of (search_documents, query_knowledge_base, list_repos)
    """

    async def search_documents(
        question: str,
        filters: dict[str, str | list[str]] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Search the knowledge base and return relevant document chunks.

        Use this tool when you need to find specific information from
        documents. Returns raw chunks with similarity scores — useful
        when you want to inspect multiple sources or compare information.

        Call list_repos first to discover what filter fields are available
        for this tenant before applying filters.

        Args:
            question: The search question or query.
            filters:  Optional metadata filters as a dictionary.
                      Keys are field names specific to this tenant.
                      Values can be a single string or list of strings.
                      Multiple values within a field are ORed.
                      Multiple fields are ANDed.

                      Examples:
                        {"application": "Smart Pricing Tool"}
                        {"application": ["SPT", "Flex Forecast"]}
                        {"table_name": "customers"}
                        {"department": "Finance", "region": "US"}
                        {"access_group": ["general", "restricted"]}

            top_k:    Number of chunks to return (1-20, default 10).

        Returns:
            Dictionary with:
              chunks:          List of relevant document chunks with scores
              total:           Number of chunks found
              question:        The original question
              skipped_filters: Filter fields not supported by any repo
        """
        log.info(
            "mcp.search_documents",
            tenant_id=tenant_id,
            question=question[:80],
            filters=filters,
            top_k=top_k,
        )

        search_filters = _build_filters(filters)

        try:
            chunks, skipped = await retrieval_service.search(
                question=question,
                tenant_id=tenant_id,
                filters=search_filters,
                top_k=top_k,
            )

            return {
                "question": question,
                "total": len(chunks),
                "skipped_filters": skipped,
                "chunks": [
                    {
                        "text": c.get("text", ""),
                        "score": round(c.get("score", 0.0), 4),
                        "file_name": c.get("file_name", ""),
                        "source_url": c.get("source_url", ""),
                        "repo_id": str(c.get("repo_id", "")),
                    }
                    for c in chunks
                ],
            }
        except Exception as e:
            log.error("mcp.search_documents.error", tenant_id=tenant_id, error=str(e))
            return {
                "error": str(e),
                "question": question,
                "total": 0,
                "chunks": [],
            }

    async def query_knowledge_base(
        question: str,
        filters: dict[str, str | list[str]] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Ask a question and get a direct answer from the knowledge base.

        Use this tool when you need a direct answer to a question.
        Searches the knowledge base, retrieves relevant context,
        and uses an LLM to generate a grounded, accurate answer.

        Call list_repos first to discover what filter fields are available
        for this tenant before applying filters.

        Args:
            question: The question to answer.
            filters:  Optional metadata filters as a dictionary.
                      Keys are field names specific to this tenant.
                      Values can be a single string or list of strings.
                      Multiple values within a field are ORed.
                      Multiple fields are ANDed.

                      Examples:
                        {"application": "Smart Pricing Tool"}
                        {"application": ["SPT", "Flex Forecast"]}
                        {"table_name": "customers"}
                        {"access_group": ["general", "restricted"]}

            top_k:    Number of context chunks to use (1-20, default 10).

        Returns:
            Dictionary with:
              answer:       Direct answer grounded in retrieved documents
              sources:      Source documents used to generate the answer
              total_chunks: Number of chunks used as context
              question:     The original question
        """
        log.info(
            "mcp.query_knowledge_base",
            tenant_id=tenant_id,
            question=question[:80],
            filters=filters,
            top_k=top_k,
        )

        search_filters = _build_filters(filters)

        try:
            # Step 1 — retrieve chunks
            chunks, _ = await retrieval_service.search(
                question=question,
                tenant_id=tenant_id,
                filters=search_filters,
                top_k=top_k,
            )

            if not chunks:
                return {
                    "question": question,
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "total_chunks": 0,
                }

            # Step 2 — resolve tenant LLM config
            tenant = await retrieval_service._tenant_repo.get_by_id(tenant_id)
            api_cfg = await resolve_api_config(
                tenant_api_config=tenant.get("api_config") if tenant else None,
                tenant_ingestion_defaults=(
                    tenant.get("ingestion_defaults", {}) if tenant else {}
                ),
            )

            # Step 3 — generate answer
            llm_svc = LLMService()
            answer = await llm_svc.generate_answer(
                question=question,
                chunks=chunks,
                api_cfg=api_cfg,
            )

            return {
                "question": question,
                "answer": answer,
                "total_chunks": len(chunks),
                "sources": [
                    {
                        "file_name": c.get("file_name", ""),
                        "source_url": c.get("source_url", ""),
                        "score": round(c.get("score", 0.0), 4),
                        "text": c.get("text", "")[:300],
                    }
                    for c in chunks
                ],
            }

        except Exception as e:
            log.error(
                "mcp.query_knowledge_base.error", tenant_id=tenant_id, error=str(e)
            )
            return {
                "error": str(e),
                "question": question,
                "answer": "An error occurred while querying the knowledge base.",
                "sources": [],
                "total_chunks": 0,
            }

    async def list_repos() -> dict[str, Any]:
        """
        List all available knowledge base repositories for this tenant.

        Use this tool to discover what knowledge bases are available
        and what filter fields you can use before searching.

        The filterable_fields show what keys you can pass in the
        filters parameter of search_documents and query_knowledge_base.

        The extractable_fields show which fields the system can
        automatically infer from the question text without explicit filters.

        Returns:
            Dictionary with:
              repos:  List of repositories with their filter fields
              total:  Number of available repositories
        """
        log.info("mcp.list_repos", tenant_id=tenant_id)

        try:
            repos = await retrieval_service._repo_repo.list_for_tenant(tenant_id)

            return {
                "tenant_id": tenant_id,
                "total": len(repos),
                "repos": [
                    {
                        "repo_id": str(r["_id"]),
                        "name": r.get("name", ""),
                        "source_type": r.get("source_type", ""),
                        "filterable_fields": r.get("retrieval_config", {}).get(
                            "filterable_fields", []
                        ),
                        "extractable_fields": r.get("retrieval_config", {}).get(
                            "extractable_fields", []
                        ),
                    }
                    for r in repos
                ],
            }

        except Exception as e:
            log.error("mcp.list_repos.error", tenant_id=tenant_id, error=str(e))
            return {"error": str(e), "tenant_id": tenant_id, "total": 0, "repos": []}

    return search_documents, query_knowledge_base, list_repos


def _build_filters(
    filters: dict[str, str | list[str]] | None,
) -> SearchFilters:
    """
    Build SearchFilters from the generic filters dict.

    Normalises each value to list[str] — SearchFilters expects
    dict[str, list[str]] but tool accepts str for convenience.

    Examples:
      {"application": "SPT"}                    → {"application": ["SPT"]}
      {"application": ["SPT", "Flex Forecast"]} → {"application": ["SPT", "Flex Forecast"]}
      {"access_group": ["general","restricted"]} → {"access_group": ["general","restricted"]}
      None                                       → {}
    """
    if not filters:
        return SearchFilters(filters={})

    normalised: dict[str, list[str]] = {}

    for field, value in filters.items():
        if not field or not value:
            continue
        if isinstance(value, str):
            normalised[field] = [value]
        elif isinstance(value, list):
            # Filter out empty strings
            clean = [v for v in value if v and isinstance(v, str)]
            if clean:
                normalised[field] = clean

    return SearchFilters(filters=normalised)
