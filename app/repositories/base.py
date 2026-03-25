"""
Base Repository — same pattern as ingestion orchestrator.

All repository classes inherit from this. The database handle is injected
via __init__ — repositories never call get_database() themselves.
This makes them fully testable: pass a mock DB, test without real MongoDB.

Dependency injection chain:
    get_database() → injected into repository __init__
    repository → injected into service __init__
    service → injected into API route via FastAPI Depends()
"""

from abc import ABC
from pymongo import AsyncDatabase


class BaseRepository(ABC):
    """
    Base class for all MongoDB repositories in the retrieval service.

    Subclasses get self.db injected — they define their own
    self.collection = self.db[COLLECTION] in __init__.
    """

    def __init__(self, db: AsyncDatabase) -> None:
        self.db = db
