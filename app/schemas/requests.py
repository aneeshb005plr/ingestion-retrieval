"""
REST API request schemas for the Retrieval Service.

SearchRequest  — POST /api/v1/{tenant_id}/search
QueryRequest   — POST /api/v1/{tenant_id}/query
"""

from pydantic import BaseModel, Field
from app.schemas.search import SearchFilters


class SearchRequest(BaseModel):
    """
    Request body for POST /api/v1/{tenant_id}/search.
    Returns ranked chunks only — no LLM call.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search question or query.",
        examples=["What is the leave policy for contractors?"],
    )
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Optional metadata filters scoped to the tenant's field names.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to return across all repos.",
    )
    repo_ids: list[str] | None = Field(
        default=None,
        description="Scope search to specific repo IDs. None = all active repos.",
    )


class QueryRequest(BaseModel):
    """
    Request body for POST /api/v1/{tenant_id}/query.
    Returns LLM-generated answer + source chunks (RAG).
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer using retrieved documents.",
        examples=["How many days of annual leave do I get?"],
    )
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description="Optional metadata filters scoped to the tenant's field names.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context.",
    )
    repo_ids: list[str] | None = Field(
        default=None,
        description="Scope search to specific repo IDs. None = all active repos.",
    )
