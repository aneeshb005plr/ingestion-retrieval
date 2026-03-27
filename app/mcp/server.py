"""
MCP Server — per-tenant MCP via registry + proxy + combined lifespans.

Key requirement (per FastMCP docs):
  "Always pass the lifespan context when mounting MCP servers"
  mcp_app.lifespan MUST be entered for session manager to initialise.

Architecture:
  1. build_mcp_apps() — builds all tenant mcp_apps at startup
     Returns: list of mcp_apps with their lifespans
     Called BEFORE FastAPI() is created

  2. get_combined_lifespan() — combines all mcp_app lifespans +
     the app's own lifespan (MongoDB, singletons) using combine_lifespans
     Passed as lifespan= to FastAPI()

  3. MCPProxy mounted at /mcp — routes requests to registry by tenant_id
     Registry populated during combined lifespan startup

  4. activate_tenant() / deactivate_tenant() — runtime management
     activate: builds new mcp_app, enters lifespan manually, adds to registry
     deactivate: removes from registry (lifespan stays running until shutdown)

Note on dynamic activate after startup:
  For tenants activated after startup, we use a background task approach
  — the mcp_app lifespan is entered within the activation coroutine
  using anyio task scope. See activate_tenant() for details.

Endpoints:
  /mcp/{tenant_id}/mcp        ← MCP tool calls
  /admin/mcp/tenants          ← list active tenants
  /admin/mcp/{id}/activate    ← activate tenant
  /admin/mcp/{id}/deactivate  ← deactivate tenant
"""

import contextlib
import structlog
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.utilities.lifespan import combine_lifespans

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

# Store mcp_apps built at startup for lifespan management
_startup_mcp_apps: list = []


async def build_startup_mcp_apps() -> list:
    """
    Build mcp_app for every active tenant.
    Called BEFORE FastAPI() is created so lifespans can be combined.

    Returns list of (tenant_id, mcp_app) tuples.
    Also populates registry for proxy routing.
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

    global _startup_mcp_apps
    _startup_mcp_apps = apps
    return apps


def get_combined_lifespan(app_lifespan):
    """
    Combine app lifespan with all tenant mcp_app lifespans.

    Per FastMCP docs:
      from fastmcp.utilities.lifespan import combine_lifespans
      app = FastAPI(lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan))

    Called after build_startup_mcp_apps() to get all mcp_app lifespans.
    """
    mcp_lifespans = [mcp_app.lifespan for _, mcp_app in _startup_mcp_apps]

    if not mcp_lifespans:
        # No active tenants — just use app lifespan
        return app_lifespan

    return combine_lifespans(app_lifespan, *mcp_lifespans)


def mount_proxy(app: FastAPI) -> None:
    """Mount MCPProxy at /mcp. Called after FastAPI app is created."""
    app.mount("/mcp", MCPProxy())
    log.info("mcp.proxy_mounted", path="/mcp")


async def activate_tenant(tenant_id: str, db=None) -> None:
    """
    Build and register a tenant MCP app at runtime (after startup).
    For post-startup activation, we mount the mcp_app directly — the
    lifespan is managed by the mcp_app itself when stateless_http=True.

    Note: For stateless_http=True, the session manager initialises
    per-request rather than requiring a persistent lifespan context.
    This is why dynamic activation works without combine_lifespans.
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

    # stateless_http=True — no persistent session state needed
    mcp_app = mcp.http_app(path="/mcp", stateless_http=True)
    return mcp_app
