"""
MCP Tools — the 3 tools exposed to AI agents.

Each tool is a plain async function.
FastMCP reads:
  - Function name      → tool name
  - Docstring          → tool description (shown to AI agent)
  - Type hints         → input/output schema (auto-generated)

Why 3 tools and not 1?
  search_documents   → agent wants RAW chunks (inspect, compare, filter itself)
  query_knowledge_base → agent wants DIRECT ANSWER (most common — DocAssist)
  list_repos         → agent discovers what knowledge bases exist before searching

These map 1:1 to existing REST endpoints — no new logic, just a different
interface layer that AI agents understand natively.

Tenant scoping:
  Tools are registered per-tenant at server mount time.
  tenant_id is captured via closure — tool calls never pass tenant_id.
  Security: one MCP mount = one tenant = one data boundary.

Filter format (agent-friendly, flat):
  {
    "application": "Smart Pricing Tool",   ← single value, string
    "domain": "XLOS",
    "access_group": "general"
  }

  Deliberately simpler than the REST API dict[str, list[str]]:
  - Agents pass one value per dimension — not multi-value OR lists
  - Multi-value access control (general + restricted) handled by caller
  - Converted to SearchFilters internally before calling retrieval service
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

    Using a factory with closure keeps tool functions clean:
      - No tenant_id parameter needed in each tool
      - Tool docstrings stay agent-readable (no internal params)
      - Security: tenant is baked in, cannot be overridden by caller

    Returns:
        Tuple of (search_documents, query_knowledge_base, list_repos)
        to be registered with FastMCP.
    """

    async def search_documents(
        question: str,
        top_k: int = 10,
        application: str | list[str] | None = None,
        domain: str | list[str] | None = None,
        access_group: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search the knowledge base and return relevant document chunks.

        Use this tool when you need to find specific information from
        documents. Returns raw chunks with similarity scores — useful
        when you want to inspect multiple sources or compare information.

        Args:
            question:     The search question or query.
            top_k:        Number of chunks to return (1–20, default 10).
            application:  Filter by application name. Single value or list.
                          e.g. "Smart Pricing Tool"
                          or   ["Smart Pricing Tool", "Flex Forecast"]
                          Use list when comparing across multiple applications.
            domain:       Filter by domain. Single or list.
                          e.g. "XLOS" or ["XLOS", "EIT"]
            access_group: Filter by access group. Single or list.
                          e.g. "general" or ["general", "restricted"]
                          Use list for privileged users who can see
                          both general and restricted documents.

        Returns:
            Dictionary with:
              chunks:    List of relevant document chunks with scores
              total:     Number of chunks found
              question:  The original question
        """
        log.info(
            "mcp.search_documents",
            tenant_id=tenant_id,
            question=question[:80],
            top_k=top_k,
            application=application,
            domain=domain,
        )

        filters = _build_filters(application, domain, access_group)

        try:
            chunks, skipped = await retrieval_service.search(
                question=question,
                tenant_id=tenant_id,
                filters=filters,
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
            return {"error": str(e), "question": question, "total": 0, "chunks": []}

    async def query_knowledge_base(
        question: str,
        top_k: int = 10,
        application: str | list[str] | None = None,
        domain: str | list[str] | None = None,
        access_group: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Ask a question and get a direct answer from the knowledge base.

        Use this tool when you need a direct answer to a question.
        Searches the knowledge base, retrieves relevant context,
        and uses an LLM to generate a grounded, accurate answer.

        Args:
            question:     The question to answer.
            top_k:        Number of context chunks to use (1–20, default 10).
            application:  Filter by application name. Single value or list.
                          e.g. "Smart Pricing Tool"
                          or   ["Smart Pricing Tool", "Flex Forecast"]
                          Use list when question spans multiple applications.
            domain:       Filter by domain. Single or list.
                          e.g. "XLOS" or ["XLOS", "EIT"]
            access_group: Filter by access group. Single or list.
                          e.g. "general" or ["general", "restricted"]
                          Pass list for privileged users who can see
                          both general and restricted documents.

        Returns:
            Dictionary with:
              answer:  Direct answer to the question
              sources: List of source documents used to generate the answer
              total_chunks: Number of chunks used as context
        """
        log.info(
            "mcp.query_knowledge_base",
            tenant_id=tenant_id,
            question=question[:80],
            top_k=top_k,
            application=application,
        )

        filters = _build_filters(application, domain, access_group)

        try:
            # Step 1 — retrieve chunks
            chunks, _ = await retrieval_service.search(
                question=question,
                tenant_id=tenant_id,
                filters=filters,
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
                        "text": c.get("text", "")[:300],  # preview only
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
        before searching. Returns repository names, types, and the
        filter fields you can use to narrow searches.

        Returns:
            Dictionary with:
              repos:  List of available knowledge base repositories
              total:  Number of repositories
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
    application: str | list[str] | None,
    domain: str | list[str] | None,
    access_group: str | list[str] | None,
) -> SearchFilters:
    """
    Build SearchFilters from tool parameters.

    Accepts single string or list of strings per dimension.
    SearchFilters expects dict[str, list[str]] — always normalise to list.

    Examples:
      application="SPT"                         → {"application": ["SPT"]}
      application=["SPT", "Flex Forecast"]      → {"application": ["SPT", "Flex Forecast"]}
      access_group=["general", "restricted"]    → {"access_group": ["general", "restricted"]}

    Within a field: values are ORed (match ANY).
    Across fields: fields are ANDed (match ALL dimensions).
    """
    raw: dict[str, list[str]] = {}

    def _to_list(val: str | list[str] | None) -> list[str]:
        """Normalise str | list[str] | None → list[str]."""
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        return [v for v in val if v]  # filter empty strings

    app_list = _to_list(application)
    dom_list = _to_list(domain)
    ag_list = _to_list(access_group)

    if app_list:
        raw["application"] = app_list
    if dom_list:
        raw["domain"] = dom_list
    if ag_list:
        raw["access_group"] = ag_list

    return SearchFilters(filters=raw)
