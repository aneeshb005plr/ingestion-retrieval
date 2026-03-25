"""
TenantRepository — read-only access to the tenants collection.

Reads:
  - api_config          → encrypted genai_api_key, base_url, llm_model
  - ingestion_defaults  → embedding_model (must match ingestion model)
  - metadata_schema     → custom fields (for context)
"""

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.repositories.base import BaseRepository
from app.core.exceptions import TenantNotFoundError, TenantInactiveError

log = structlog.get_logger(__name__)


class TenantRepository(BaseRepository):

    COLLECTION = "tenants"

    def __init__(self, db: AsyncDatabase) -> None:
        super().__init__(db)
        self.collection = self.db[self.COLLECTION]

    async def get_by_id(self, tenant_id: str) -> dict:
        """
        Fetch tenant document by ID.

        Raises:
            TenantNotFoundError  — tenant_id does not exist
            TenantInactiveError  — tenant exists but is_active=False
        """
        doc = await self.collection.find_one({"_id": tenant_id})

        if not doc:
            raise TenantNotFoundError(f"Tenant '{tenant_id}' not found.")
        if not doc.get("is_active", True):
            raise TenantInactiveError(f"Tenant '{tenant_id}' is inactive.")

        log.debug("tenant_repo.fetched", tenant_id=tenant_id)
        return doc
