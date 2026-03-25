"""
Domain exceptions for the Retrieval Service.
"""


class TenantNotFoundError(Exception):
    """Raised when tenant_id does not exist in MongoDB."""

    pass


class TenantInactiveError(Exception):
    """Raised when tenant exists but is_active=False."""

    pass


class NoActiveReposError(Exception):
    """Raised when tenant has no active repos to search."""

    pass


class EmbeddingError(Exception):
    """Raised when question embedding fails."""

    pass


class VectorSearchError(Exception):
    """Raised when MongoDB $vectorSearch fails."""

    pass


class LLMError(Exception):
    """Raised when LLM answer generation fails."""

    pass
