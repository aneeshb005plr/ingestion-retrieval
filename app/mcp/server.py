"""
MCP Server — mounts per-tenant MCP instances on the FastAPI app.

Architecture:
  One FastMCP instance per active tenant, mounted at /mcp/{tenant_id}.

  Why per-tenant?
    Security:  One tenant cannot call another tenant's tools.
    Scoping:   tenant_id baked into tools via closure — never in tool params.
    Discovery: Copilot Studio points to /mcp/docassist_dev and sees only
               that tenant's tools and repos.

Transport:
  Streamable HTTP (POST /mcp/{tenant_id}).
  This is the current MCP standard as of 2025 — SSE deprecated Apr 1 2026.
  FastMCP handles the transport layer — we just define tools.

Mounting:
  Called from main.py lifespan on startup.
  Reads all active tenants from MongoDB.
  Creates one FastMCP instance per tenant.
  Mounts each as a sub-application at /mcp/{tenant_id}.

Adding a new tenant:
  Restart the service — lifespan re-mounts all active tenants.
  Future: hot-reload endpoint to mount new tenants without restart.

Example URLs:
  /mcp/docassist_dev   ← DocAssist tenant
  /mcp/smartquery_dev  ← SmartQuery tenant

Copilot Studio registration:
  URL: https://your-service.com/mcp/docassist_dev
  Auth: bearer token or API key (handled by Ocelot upstream)
"""

import structlog
from fastapi import FastAPI
from fastmcp import FastMCP

from app.db.mongo import get_database
from app.repositories.tenant_repo import TenantRepository
from app.repositories.repo_repo import RepoRepository
from app.api.dependencies import (
    get_vector_store,
    get_filter_builder,
    get_embedding_factory,
    get_filter_extractor,
)
from app.services.retrieval_service import RetrievalService
from app.mcp.tools import make_tools

log = structlog.get_logger(__name__)


async def mount_mcp_servers(app: FastAPI) -> list[str]:
    """
    Mount one FastMCP instance per active tenant.

    Called from main.py lifespan AFTER singletons are initialised.
    Returns list of mounted tenant_ids for logging.

    Each mount:
      1. Creates a RetrievalService scoped to that tenant
      2. Calls make_tools() to get the 3 tool functions
      3. Registers tools with FastMCP
      4. Mounts FastMCP as ASGI sub-app at /mcp/{tenant_id}
    """
    db = get_database()
    tenant_repo = TenantRepository(db)

    tenants = await tenant_repo.list_active()
    mounted = []

    for tenant in tenants:
        tenant_id = tenant["_id"]
        try:
            _mount_tenant(app, tenant_id, db)
            mounted.append(tenant_id)
            log.info("mcp.tenant_mounted", tenant_id=tenant_id)
        except Exception as e:
            # Don't fail startup if one tenant fails to mount
            log.error(
                "mcp.tenant_mount_failed",
                tenant_id=tenant_id,
                error=str(e),
            )

    log.info("mcp.servers_ready", count=len(mounted), tenants=mounted)
    return mounted


def _mount_tenant(app: FastAPI, tenant_id: str, db) -> None:
    """
    Create and mount FastMCP for a single tenant.
    """
    # Build RetrievalService for this tenant
    # Uses same singletons as REST routes — no duplicate connections
    repo_repo = RepoRepository(db)
    retrieval_service = RetrievalService(
        tenant_repo=TenantRepository(db),
        repo_repo=repo_repo,
        vector_store=get_vector_store(),
        filter_builder=get_filter_builder(),
        embedding_factory=get_embedding_factory(),
        filter_extractor=get_filter_extractor(),
    )

    # Create FastMCP instance for this tenant
    mcp = FastMCP(
        name=f"Vector Platform — {tenant_id}",
        instructions=(
            f"Knowledge base assistant for tenant '{tenant_id}'. "
            f"Use search_documents to find relevant chunks, "
            f"query_knowledge_base to get direct answers, "
            f"or list_repos to discover available knowledge bases."
        ),
    )

    # Register the 3 tools — scoped to this tenant via closure
    search_fn, query_fn, list_fn = make_tools(tenant_id, retrieval_service)
    mcp.tool()(search_fn)
    mcp.tool()(query_fn)
    mcp.tool()(list_fn)

    # Mount as ASGI sub-app at /mcp/{tenant_id}
    # Streamable HTTP transport — compatible with Copilot Studio
    app.mount(
        f"/mcp/{tenant_id}",
        mcp.streamable_http_app(),
    )

    log.debug(
        "mcp.tenant_instance_created",
        tenant_id=tenant_id,
        path=f"/mcp/{tenant_id}",
    )
