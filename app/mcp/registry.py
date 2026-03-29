"""
TenantMCPRegistry — in-memory registry of active tenant MCP apps.

Design: stateless_http=True means each tool call is independent.
No session state, no persistent connections, no lifespan complexity.

Activate:   registry["tenant_id"] = mcp_app  ← instant
Deactivate: del registry["tenant_id"]         ← instant, no cleanup needed

Why stateless is correct for this platform:
  search_documents()       → one call → one response → done
  query_knowledge_base()   → one call → one response → done
  list_repos()             → one call → one response → done

  None of these need session state between calls.
  Each call hits MongoDB fresh.
  No long-lived connections to manage.

Thread safety:
  asyncio is single-threaded — plain dict operations are atomic
  in CPython. All methods are synchronous. No lock needed.
"""

import structlog

log = structlog.get_logger(__name__)


class TenantMCPRegistry:

    def __init__(self) -> None:
        self._apps: dict[str, object] = {}

    def activate(self, tenant_id: str, mcp_app) -> None:
        """
        Add tenant MCP app to registry.
        Replaces existing entry if tenant already active.
        Synchronous — no lifespan to enter for stateless apps.
        """
        self._apps[tenant_id] = mcp_app
        log.info("mcp.registry.activated", tenant_id=tenant_id)

    def deactivate(self, tenant_id: str) -> bool:
        """
        Remove tenant from registry.
        Returns True if found, False if not active.
        Synchronous — no lifespan to exit for stateless apps.
        """
        if tenant_id not in self._apps:
            log.warning("mcp.registry.not_found", tenant_id=tenant_id)
            return False
        del self._apps[tenant_id]
        log.info("mcp.registry.deactivated", tenant_id=tenant_id)
        return True

    def get(self, tenant_id: str):
        """Return mcp_app for tenant or None if not active."""
        return self._apps.get(tenant_id)

    def list_active(self) -> list[str]:
        """Return list of currently active tenant IDs."""
        return list(self._apps.keys())

    def clear(self) -> None:
        """Clear all tenants — called on shutdown."""
        count = len(self._apps)
        self._apps.clear()
        log.info("mcp.registry.cleared", count=count)


# Global singleton
registry = TenantMCPRegistry()
