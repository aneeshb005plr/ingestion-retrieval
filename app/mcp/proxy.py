"""
MCPProxy — single ASGI app mounted at /mcp that routes to tenant registry.

How it works:
  Request: POST /mcp/docassist_dev/mcp
           │
           ▼
  Starlette Mount("/mcp", proxy):
    scope["path"]      = "/mcp/docassist_dev/mcp"  ← full path preserved
    scope["root_path"] = "/mcp"                    ← mount point

  MCPProxy strips root_path prefix:
    tenant_path = "/docassist_dev/mcp"
    tenant_id   = "docassist_dev"
    remaining   = "/mcp"

  Looks up registry["docassist_dev"] → mcp_app
  Forwards request with path="/mcp" → mcp_app handles tool call

Important:
  Starlette Mount does NOT strip the prefix from scope["path"].
  It only sets scope["root_path"].
  We must strip root_path manually before parsing tenant_id.

Inactive tenant (404):
  registry.get("inactive_tenant") → None
  proxy returns: 404 { "detail": "Tenant not found or inactive" }
"""

import json
import structlog

from app.mcp.registry import registry

log = structlog.get_logger(__name__)


class MCPProxy:
    """
    ASGI middleware that routes /mcp/{tenant_id}/... to the
    correct tenant MCP app from the registry.

    Mount once at startup:
        app.mount("/mcp", MCPProxy())

    Never needs to change — registry handles activate/deactivate.
    """

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            return

        # Starlette Mount sets root_path but does NOT strip it from path.
        # Full path is preserved in scope["path"].
        #
        # e.g. app.mount("/mcp", proxy)
        #      Request: POST /mcp/docassist_dev/mcp
        #      scope["path"]      = "/mcp/docassist_dev/mcp"  ← full path
        #      scope["root_path"] = "/mcp"                    ← mount point
        #
        # We must strip root_path prefix ourselves to get the tenant path.
        full_path: str = scope.get("path", "/")
        root_path: str = scope.get("root_path", "")

        # Strip root_path prefix → get /docassist_dev/mcp
        if root_path and full_path.startswith(root_path):
            tenant_path = full_path[len(root_path) :]
        else:
            tenant_path = full_path

        # Parse tenant_id from /docassist_dev/mcp
        # → parts = ["docassist_dev", "mcp"]
        parts = tenant_path.strip("/").split("/", 1)

        if not parts or not parts[0]:
            await self._not_found(send, "Missing tenant_id in path")
            return

        tenant_id = parts[0]
        remaining = "/" + parts[1] if len(parts) > 1 else "/"

        mcp_app = registry.get(tenant_id)

        if mcp_app is None:
            log.warning("mcp.proxy.tenant_not_found", tenant_id=tenant_id)
            await self._not_found(send, f"Tenant '{tenant_id}' not found or inactive")
            return

        log.debug(
            "mcp.proxy.forwarding",
            tenant_id=tenant_id,
            path=remaining,
        )

        # Forward to tenant's mcp_app with corrected path
        forwarded_scope = {
            **scope,
            "path": remaining,
            "raw_path": remaining.encode(),
            # root_path helps mcp_app know where it's mounted
            "root_path": scope.get("root_path", "") + f"/mcp/{tenant_id}",
        }

        await mcp_app(forwarded_scope, receive, send)

    @staticmethod
    async def _not_found(send, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )
