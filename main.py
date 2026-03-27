"""
Retrieval Service — entry point.

Structure:
  1. lifespan()    — standalone async context manager
  2. create_app()  — app factory, accepts environment config
  3. app           — module-level instance via create_app()
  4. __main__      — uvicorn runner

MCP startup order (critical):
  1. connect_to_mongo() + init_singletons()
  2. build_startup_mcp_apps() — reads active tenants, builds mcp_apps
  3. Enter each mcp_app.lifespan via AsyncExitStack
  4. yield (service ready)
  5. shutdown on exit

MCP endpoints:
  /mcp/{tenant_id}/mcp       ← MCP tool endpoint
  /admin/mcp/tenants         ← list active tenants
  /admin/mcp/{id}/activate   ← activate tenant MCP
  /admin/mcp/{id}/deactivate ← deactivate (no restart needed)
"""

import contextlib
import os
import structlog

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongo import connect_to_mongo, close_mongo
from app.api.dependencies import init_singletons, shutdown_singletons
from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.api.mcp_admin import router as mcp_admin_router
from app.mcp.server import (
    build_startup_mcp_apps,
    mount_proxy,
    shutdown_mcp,
)

log = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — startup and shutdown sequence.

    Startup:
      1. Logging + MongoDB + singletons
      2. Build tenant mcp_apps (needs singletons ready)
      3. Enter each mcp_app.lifespan via AsyncExitStack
         (required by FastMCP for session manager initialisation)
      4. yield → service is ready

    Shutdown:
      5. AsyncExitStack exits all mcp_app lifespans
      6. MCP registry cleared
      7. Singletons + MongoDB closed
    """
    # ── Startup ───────────────────────────────────────────────────
    setup_logging()
    log.info(
        "retrieval_service.starting",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )

    await connect_to_mongo()
    init_singletons()

    # Build tenant MCP apps — must run after init_singletons()
    # so VectorStore and FilterBuilder are available
    tenant_apps = await build_startup_mcp_apps()
    log.info("mcp.apps_built", count=len(tenant_apps))

    # Enter each mcp_app lifespan
    # Per FastMCP docs: lifespan must be entered for session manager to init
    # mcp_app.lifespan(mcp_app) returns AsyncContextManager directly —
    # do NOT wrap with asynccontextmanager again
    async with contextlib.AsyncExitStack() as stack:
        for tenant_id, mcp_app in tenant_apps:
            try:
                await stack.enter_async_context(mcp_app.lifespan(mcp_app))
                log.info("mcp.lifespan_entered", tenant_id=tenant_id)
            except Exception as e:
                log.error(
                    "mcp.lifespan_enter_failed",
                    tenant_id=tenant_id,
                    error=str(e),
                )

        log.info("retrieval_service.ready")
        yield

    # ── Shutdown ──────────────────────────────────────────────────
    shutdown_mcp()
    shutdown_singletons()
    await close_mongo()
    log.info("retrieval_service.stopped")


def create_app(environment_config: str = "development") -> FastAPI:
    """
    Application factory.

    Args:
        environment_config: environment name (development/production).
                            Controls docs_url visibility.

    Returns:
        Configured FastAPI instance with all routers and MCP proxy mounted.
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/scalar" if environment_config != "production" else None,
        redoc_url=None,
    )

    # ── MCP proxy — must mount before routes ──────────────────────
    # Single proxy at /mcp routes to per-tenant mcp_apps in registry
    mount_proxy(application)

    # ── REST routes ───────────────────────────────────────────────
    application.include_router(health_router)
    application.include_router(api_router)
    application.include_router(mcp_admin_router)

    return application


# ── App instance ──────────────────────────────────────────────────
environment_name = os.environ.get("FASTAPI_CONFIG", "development")
app = create_app(environment_name)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=True,
    )
