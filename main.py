"""
Retrieval Service — entry point.

Startup sequence:
  1. Setup structured logging
  2. Connect to MongoDB Atlas
  3. Register REST API routes
  4. MCP servers mounted per-tenant in Phase R4

Run locally:
  uvicorn main:app --reload --port 8001

Kubernetes:
  Port is container-internal — Ocelot/ingress handles external routing.
"""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongo import connect_to_mongo, close_mongo
from app.api.dependencies import init_singletons, shutdown_singletons
from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.mcp.server import mount_mcp_servers

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────
    setup_logging()
    log.info(
        "retrieval_service.starting",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )
    await connect_to_mongo()
    init_singletons()  # VectorStoreProvider + FilterBuilder
    # Mount one MCP server per active tenant at /mcp/{tenant_id}
    # Must run AFTER init_singletons() — tools need vector store + filter builder
    await mount_mcp_servers(app)
    log.info("retrieval_service.ready")
    yield
    # ── Shutdown ──────────────────────────────────────────────────
    shutdown_singletons()
    await close_mongo()
    log.info("retrieval_service.stopped")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/scalar" if settings.ENVIRONMENT != "production" else None,
    redoc_url=None,
)

# ── Routes ────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(api_router)
