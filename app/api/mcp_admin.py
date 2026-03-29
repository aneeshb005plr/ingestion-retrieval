"""
MCP Admin API — manage tenant MCP servers without restart.

Endpoints:
  GET  /admin/mcp/tenants              ← list active tenants in registry
  POST /admin/mcp/{tenant_id}/activate ← activate tenant MCP
  POST /admin/mcp/{tenant_id}/deactivate ← deactivate tenant MCP

Who calls these:
  - Orchestrator: when tenant is_active changes via PATCH /tenants/{id}
  - Ops team: manual activate/deactivate
  - Health check: verify expected tenants are active

Security:
  These are admin endpoints — should be protected by Ocelot
  with admin role check in production.
  For now: accessible only within the cluster (not exposed externally).
"""

import structlog
from fastapi import APIRouter, HTTPException, status

from app.mcp.registry import registry
from app.mcp.server import activate_tenant, deactivate_tenant

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/mcp", tags=["mcp-admin"])


@router.get(
    "/tenants",
    summary="List active MCP tenants",
    description="Returns all tenant IDs currently active in the MCP registry.",
)
async def list_active_tenants() -> dict:
    tenants = registry.list_active()
    return {
        "active_tenants": tenants,
        "count": len(tenants),
    }


@router.post(
    "/{tenant_id}/activate",
    summary="Activate MCP for a tenant",
    description=(
        "Build and register a tenant's MCP server. "
        "Safe to call on already-active tenant — replaces cleanly. "
        "Called when tenant is_active → True."
    ),
)
async def activate_tenant_mcp(tenant_id: str) -> dict:
    try:
        await activate_tenant(tenant_id)
        log.info("mcp.admin.activated", tenant_id=tenant_id)
        return {
            "tenant_id": tenant_id,
            "status": "activated",
            "endpoint": f"/mcp/{tenant_id}/mcp",
        }
    except Exception as e:
        log.error("mcp.admin.activate_failed", tenant_id=tenant_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate MCP for tenant '{tenant_id}': {e}",
        )


@router.post(
    "/{tenant_id}/deactivate",
    summary="Deactivate MCP for a tenant",
    description=(
        "Remove a tenant from the MCP registry and shut down its session manager. "
        "In-flight requests complete normally. New requests get 404. "
        "Called when tenant is_active → False."
    ),
)
async def deactivate_tenant_mcp(tenant_id: str) -> dict:
    found = deactivate_tenant(tenant_id)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' is not active in MCP registry.",
        )
    log.info("mcp.admin.deactivated", tenant_id=tenant_id)
    return {
        "tenant_id": tenant_id,
        "status": "deactivated",
    }
