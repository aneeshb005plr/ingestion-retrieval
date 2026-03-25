"""
Health check endpoints.

GET /health/live   — liveness: is the process running?
GET /health/ready  — readiness: is MongoDB reachable?
"""

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.db.mongo import get_database

log = structlog.get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health/live")
async def liveness() -> dict:
    return {"status": "live"}


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    try:
        db = get_database()
        await db.command("ping")
        return JSONResponse({"status": "ready", "mongodb": "ok"})
    except Exception as e:
        log.error("health.ready.failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "mongodb": "unreachable"},
        )
