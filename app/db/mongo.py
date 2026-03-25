"""
MongoDB connection for the Retrieval Service.
Same pattern as ingestion worker — single AsyncMongoClient
shared across the application lifetime via lifespan.
"""

import structlog
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import settings

log = structlog.get_logger(__name__)

_client: AsyncMongoClient | None = None
_db: AsyncDatabase | None = None


async def connect_to_mongo() -> None:
    global _client, _db
    log.info("mongodb.connecting", db=settings.MONGO_DB_NAME)
    _client = AsyncMongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tz_aware=True,
    )
    # Verify connection
    await _client.admin.command("ping")
    _db = _client[settings.MONGO_DB_NAME]
    log.info("mongodb.connected", db=settings.MONGO_DB_NAME)


async def close_mongo() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        log.info("mongodb.disconnected")


def get_database() -> AsyncDatabase:
    if _db is None:
        raise RuntimeError("MongoDB not connected. Call connect_to_mongo() first.")
    return _db
