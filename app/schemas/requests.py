"""
REST API request schemas for the Retrieval Service.

SearchRequest  — POST /api/v1/{tenant_id}/search
QueryRequest   — POST /api/v1/{tenant_id}/query

Filter model:
  must_filters   → hard access boundaries (always applied)
  should_filters → soft scope (LLM narrows if auto_extract=True)
  auto_extract   → whether platform narrows should_filters from question
"""

from pydantic import BaseModel, Field
from app.schemas.search import SearchFilters


class SearchRequest(BaseModel):
    """
    Request body for POST /api/v1/{tenant_id}/search.
    Returns ranked chunks only — no LLM call.

    Example — DocAssist user with mixed access:
    {
      "question": "Who is the owner of SPT?",
      "filters": {
        "must_filters": {
          "access_key": ["Smart Pricing Tool::general",
                         "Flex Forecast::general",
                         "Flex Forecast::restricted"]
        },
        "should_filters": {
          "application": ["Smart Pricing Tool", "Flex Forecast", "Leave App"]
        },
        "auto_extract": true
      }
    }

    Example — no access control (internal):
    {
      "question": "Who owns SPT?",
      "filters": { "must_filters": {}, "should_filters": {}, "auto_extract": true }
    }

    Example — user explicit filter from UI dropdown:
    {
      "question": "Who owns SPT?",
      "filters": {
        "must_filters": {
          "access_key": ["Smart Pricing Tool::general"],
          "application": ["Smart Pricing Tool"]   ← explicit selection
        },
        "should_filters": {},
        "auto_extract": false   ← no need, already explicit
      }
    }
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
        description=(
            "Filter model with must_filters (hard), "
            "should_filters (soft scope), and auto_extract flag."
        ),
    )
    top_k: int = Field(
        default=10,
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
        description=(
            "Filter model with must_filters (hard), "
            "should_filters (soft scope), and auto_extract flag."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context.",
    )
    repo_ids: list[str] | None = Field(
        default=None,
        description="Scope search to specific repo IDs. None = all active repos.",
    )
