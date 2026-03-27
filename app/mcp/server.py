"""
MCP Server — per-tenant MCP via registry + proxy.

Key requirement (per FastMCP docs):
  "Always pass the lifespan context when mounting MCP servers"
  mcp_app.lifespan MUST be entered for session manager to initialise.

Architecture:
  1. build_startup_mcp_apps()
     Reads active tenants from MongoDB, builds one mcp_app per tenant.
     Populates registry for proxy routing.
     Returns list of (tenant_id, mcp_app) tuples.
     Called from main.py lifespan AFTER init_singletons().

  2. main.py lifespan enters each mcp_app.lifespan via AsyncExitStack:
       async with contextlib.AsyncExitStack() as stack:
           for tenant_id, mcp_app in tenant_apps:
               await stack.enter_async_context(mcp_app.lifespan(mcp_app))

  3. MCPProxy mounted at /mcp routes requests to registry by tenant_id.
     Registry populated in step 1.

  4. activate_tenant() / deactivate_tenant() — runtime management.
     deactivate: instant registry removal, no restart needed.
     activate: builds new mcp_app, adds to registry.

Endpoints:
  /mcp/{tenant_id}/mcp        ← MCP tool calls
  /admin/mcp/tenants          ← list active tenants
  /admin/mcp/{id}/activate    ← activate tenant
  /admin/mcp/{id}/deactivate  ← deactivate tenant
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
from app.mcp.registry import registry
from app.mcp.proxy import MCPProxy

log = structlog.get_logger(__name__)


async def build_startup_mcp_apps() -> list[tuple[str, object]]:
    """
    Build mcp_app for every active tenant.

    Called from main.py lifespan after init_singletons() so
    VectorStore and FilterBuilder are available.

    Returns:
        list of (tenant_id, mcp_app) tuples.
        mcp_app lifespans are entered by main.py lifespan
        via AsyncExitStack.
    """
    db = get_database()
    tenant_repo = TenantRepository(db)
    tenants = await tenant_repo.list_active()

    apps = []
    for tenant in tenants:
        tenant_id = tenant["_id"]
        try:
            mcp_app = _build_tenant_mcp_app(tenant_id, db)
            registry.activate(tenant_id, mcp_app)
            apps.append((tenant_id, mcp_app))
            log.info("mcp.tenant_built", tenant_id=tenant_id)
        except Exception as e:
            log.error("mcp.tenant_build_failed", tenant_id=tenant_id, error=str(e))

    return apps


def mount_proxy(app: FastAPI) -> None:
    """Mount MCPProxy at /mcp. Called from create_app() in main.py."""
    app.mount("/mcp", MCPProxy())
    log.info("mcp.proxy_mounted", path="/mcp")


async def activate_tenant(tenant_id: str, db=None) -> None:
    """
    Build and register a tenant MCP app at runtime (after startup).
    Called from admin API when a tenant is re-activated.

    Note: post-startup activations do not have their lifespan entered
    via AsyncExitStack. stateless_http=True means the session manager
    initialises per-request — no persistent lifespan context needed
    for dynamic activations.
    """
    if db is None:
        db = get_database()

    mcp_app = _build_tenant_mcp_app(tenant_id, db)
    registry.activate(tenant_id, mcp_app)

    log.info(
        "mcp.tenant_activated",
        tenant_id=tenant_id,
        endpoint=f"/mcp/{tenant_id}/mcp",
    )


def deactivate_tenant(tenant_id: str) -> bool:
    """Remove tenant from registry. Instant, no restart needed."""
    return registry.deactivate(tenant_id)


def shutdown_mcp() -> None:
    """Clear registry on shutdown."""
    registry.clear()
    log.info("mcp.shutdown_complete")


def _build_tenant_mcp_app(tenant_id: str, db) -> object:
    """Build a FastMCP ASGI app for a single tenant."""
    retrieval_service = RetrievalService(
        tenant_repo=TenantRepository(db),
        repo_repo=RepoRepository(db),
        vector_store=get_vector_store(),
        filter_builder=get_filter_builder(),
        embedding_factory=get_embedding_factory(),
        filter_extractor=get_filter_extractor(),
    )

    mcp = FastMCP(
        name=f"Vector Platform — {tenant_id}",
        instructions=(
            f"Knowledge base assistant for tenant '{tenant_id}'. "
            f"Call list_repos to discover available knowledge bases, "
            f"then use search_documents or query_knowledge_base."
        ),
    )

    search_fn, query_fn, list_fn = make_tools(tenant_id, retrieval_service)
    mcp.tool()(search_fn)
    mcp.tool()(query_fn)
    mcp.tool()(list_fn)

    # stateless_http=True — each tool call is independent, no session state
    mcp_app = mcp.http_app(path="/mcp", stateless_http=True)
    return mcp_app
