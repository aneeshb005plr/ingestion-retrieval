"""
RepoRepository — read-only access to the source_repositories collection.

Reads per repo:
  - vector_config.index_name  → Atlas index to search
  - retrieval_config          → filterable_fields, general_flag_field/value
  - source_type               → for logging/context
  - is_active                 → only active repos are searched
"""

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.repositories.base import BaseRepository
from app.core.exceptions import NoActiveReposError

log = structlog.get_logger(__name__)


class RepoRepository(BaseRepository):

    COLLECTION = "source_repositories"

    def __init__(self, db: AsyncDatabase) -> None:
        super().__init__(db)
        self.collection = self.db[self.COLLECTION]

    async def get_active_for_tenant(self, tenant_id: str) -> list[dict]:
        """
        Return all active repos for a tenant.

        Raises:
            NoActiveReposError — tenant has no active repos
        """
        cursor = self.collection.find(
            {
                "tenant_id": tenant_id,
                "is_active": True,
            }
        )
        repos = await cursor.to_list(length=None)

        if not repos:
            raise NoActiveReposError(
                f"No active repositories found for tenant '{tenant_id}'."
            )

        log.debug(
            "repo_repo.fetched",
            tenant_id=tenant_id,
            count=len(repos),
        )
        return repos

    async def get_by_id(self, repo_id: str) -> dict | None:
        """Fetch a single repo by ID. Returns None if not found."""
        return await self.collection.find_one({"_id": repo_id})

    async def get_active_by_ids(
        self,
        tenant_id: str,
        repo_ids: list[str],
    ) -> list[dict]:
        """
        Return active repos filtered to a specific list of repo_ids.
        Used when the caller scopes the search to specific repos.

        Raises:
            NoActiveReposError — none of the given repo_ids are active
        """
        cursor = self.collection.find(
            {
                "_id": {"$in": repo_ids},
                "tenant_id": tenant_id,
                "is_active": True,
            }
        )
        repos = await cursor.to_list(length=None)

        if not repos:
            raise NoActiveReposError(
                f"No active repositories found for "
                f"tenant '{tenant_id}' with ids {repo_ids}."
            )
        return repos

    async def list_for_tenant(self, tenant_id: str) -> list[dict]:
        """
        Return summary info for all active repos — used by MCP list_repos tool.
        Returns only fields needed for display, not full docs.
        """
        cursor = self.collection.find(
            {"tenant_id": tenant_id, "is_active": True},
            {
                "_id": 1,
                "name": 1,
                "source_type": 1,
                "retrieval_config": 1,
                "vector_config": 1,
            },
        )
        return await cursor.to_list(length=None)

    async def get_distinct_filter_values(
        self,
        tenant_id: str,
        field_name: str,
        repo_ids: list[str] | None = None,
        repos: list[dict] | None = None,
    ) -> list[str]:
        """
        Get distinct values for an extractable field from the correct collections.

        Three-tier collection routing:
          shared_tenant    → query "vector_store"
          dedicated_tenant → query "vector_store_{tenant_id}"
          dedicated_repo   → query "vector_store_{repo_id}"

        We must query ALL relevant collections — a tenant may have repos
        across different tiers (e.g. some shared, one dedicated).

        IMPORTANT: Only called for extractable_fields — content dimensions
        like "application" and "domain". Never access control fields.

        Args:
            tenant_id:  Tenant to scope the query
            field_name: e.g. "application" or "domain"
            repo_ids:   Optional scope to specific repos
            repos:      Repo config docs — used to resolve collection names
                        If None, falls back to default "vector_store"

        e.g. field_name="application" →
             ["Smart Pricing Tool", "Flex Forecast", "LeaveApp"]
        """
        # Build a set of (collection_name, optional repo_id_filter) to query
        # Group repos by collection so we minimise DB round trips
        collection_repo_map: dict[str, list[str]] = {}  # collection → [repo_ids]

        if repos:
            for repo in repos:
                rid = str(repo.get("_id", ""))
                if repo_ids and rid not in repo_ids:
                    continue
                vector_cfg = repo.get("vector_config", {})
                col = vector_cfg.get("collection_name") or "vector_store"
                collection_repo_map.setdefault(col, []).append(rid)
        else:
            # No repo config available — fall back to shared collection
            collection_repo_map["vector_store"] = repo_ids or []

        all_values: set[str] = set()

        for col_name, col_repo_ids in collection_repo_map.items():
            collection = self.db[col_name]
            match: dict = {"tenant_id": tenant_id, field_name: {"$exists": True}}
            if col_repo_ids:
                match["repo_id"] = {"$in": col_repo_ids}
            values = await collection.distinct(field_name, match)
            all_values.update(values)

        # Exclude empty strings and placeholder values
        # "general" is a path convention default — never a meaningful filter
        return [v for v in all_values if v and v not in ("general", "unknown")]
