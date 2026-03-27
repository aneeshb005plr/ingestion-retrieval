"""
Retrieval Service — entry point.

Startup sequence (ORDER MATTERS for MCP lifespan):
  1. Setup logging
  2. Connect to MongoDB
  3. Initialise singletons (VectorStore, FilterBuilder etc.)
  4. Build tenant mcp_apps (reads active tenants from MongoDB)
  5. Combine all lifespans (app + each mcp_app)
  6. Create FastAPI with combined lifespan
  7. Mount routes + MCPProxy

Per FastMCP docs:
  "Always pass the lifespan context when mounting MCP servers"
  combine_lifespans() handles multiple tenants cleanly.

MCP endpoints:
  /mcp/{tenant_id}/mcp       ← MCP tool endpoint
  /admin/mcp/tenants         ← list active tenants
  /admin/mcp/{id}/activate   ← activate tenant MCP
  /admin/mcp/{id}/deactivate ← deactivate (no restart)

Run:
  uvicorn main:app --reload --port 8001
"""

import contextlib
import structlog

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongo import connect_to_mongo, close_mongo
from app.api.dependencies import init_singletons, shutdown_singletons

log = structlog.get_logger(__name__)


async def _startup():
    """Run all startup tasks and return combined lifespan."""
    setup_logging()
    log.info(
        "retrieval_service.starting",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )
    await connect_to_mongo()
    init_singletons()


async def _shutdown():
    """Run all shutdown tasks."""
    from app.mcp.server import shutdown_mcp

    shutdown_mcp()
    shutdown_singletons()
    await close_mongo()
    log.info("retrieval_service.stopped")


def create_app():
    """
    Application factory.

    Builds mcp_apps synchronously-compatible by deferring async work
    to the lifespan. Uses combine_lifespans from FastMCP to ensure
    every tenant's session manager is properly initialised.
    """
    from fastapi import FastAPI
    from fastmcp.utilities.lifespan import combine_lifespans

    from app.mcp.server import (
        build_startup_mcp_apps,
        get_combined_lifespan,
        mount_proxy,
    )
    from app.api.health import router as health_router
    from app.api.routes import router as api_router
    from app.api.mcp_admin import router as mcp_admin_router

    @contextlib.asynccontextmanager
    async def app_lifespan(app: FastAPI):
        """Core app lifespan — DB + singletons."""
        await _startup()
        yield
        await _shutdown()

    @contextlib.asynccontextmanager
    async def full_lifespan(app: FastAPI):
        """
        Full lifespan: build mcp_apps → combine lifespans → run.

        We can't pre-build mcp_apps before this point because
        init_singletons() must run first (needs VectorStore ready).

        Approach:
          1. Run app_lifespan startup (DB + singletons)
          2. Build all tenant mcp_apps (needs singletons)
          3. Enter each mcp_app lifespan via AsyncExitStack
          4. yield (service is ready)
          5. Exit all lifespans in reverse
        """
        await connect_to_mongo()
        init_singletons()
        log.info("retrieval_service.singletons_ready")

        # Build tenant mcp_apps now that singletons are available
        tenant_apps = await build_startup_mcp_apps()
        log.info("mcp.apps_built", count=len(tenant_apps))

        # Enter all mcp_app lifespans via AsyncExitStack
        # mcp_app.lifespan(mcp_app) already returns AsyncContextManager
        # DO NOT wrap with asynccontextmanager again
        async with contextlib.AsyncExitStack() as stack:
            for tenant_id, mcp_app in tenant_apps:
                try:
                    await stack.enter_async_context(mcp_app.lifespan(mcp_app))
                    log.info("mcp.lifespan_entered", tenant_id=tenant_id)
                except Exception as e:
                    log.error(
                        "mcp.lifespan_enter_failed", tenant_id=tenant_id, error=str(e)
                    )

            log.info("retrieval_service.ready")
            yield

        # Shutdown
        from app.mcp.server import shutdown_mcp

        shutdown_mcp()
        shutdown_singletons()
        await close_mongo()
        log.info("retrieval_service.stopped")

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=full_lifespan,
        docs_url="/scalar" if settings.ENVIRONMENT != "production" else None,
        redoc_url=None,
    )

    # Mount proxy and routes
    mount_proxy(app)
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(mcp_admin_router)

    return app


app = create_app()
