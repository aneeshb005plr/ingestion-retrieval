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
    ) -> list[str]:
        """
        Get distinct values for an extractable field from vector_store.
        Used to give FilterExtractor (LLM) the real values to choose from —
        prevents hallucinated filter values.

        IMPORTANT: This should only be called for extractable_fields —
        content dimensions like "application" and "domain".
        The guard lives in retrieval_service._auto_extract_filters which
        only iterates over repo.retrieval_config.extractable_fields.
        Access control fields (access_group, source_id) are never passed here.

        e.g. field_name="application" →
             ["Smart Pricing Tool", "Flex Forecast", "LeaveApp"]
        e.g. field_name="domain" →
             ["XLOS", "EIT"]
        """
        vector_collection = self.db["vector_store"]
        match: dict = {"tenant_id": tenant_id, field_name: {"$exists": True}}
        if repo_ids:
            match["repo_id"] = {"$in": repo_ids}

        values = await vector_collection.distinct(field_name, match)
        # Exclude empty strings and the "general" placeholder value —
        # "general" is a path convention default, not a meaningful filter value
        # for LLM extraction (user would never ask "tell me about general docs")
        return [v for v in values if v and v not in ("general", "unknown")]
