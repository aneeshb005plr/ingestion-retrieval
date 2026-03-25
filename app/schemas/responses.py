"""
REST API response schemas for the Retrieval Service.

ChunkResponse   — single chunk returned from search
SearchResponse  — POST /api/v1/{tenant_id}/search
QueryResponse   — POST /api/v1/{tenant_id}/query
RepoResponse    — GET  /api/v1/{tenant_id}/repos
"""

from pydantic import BaseModel, Field
from typing import Any


class ChunkResponse(BaseModel):
    """
    A single retrieved document chunk with its similarity score.
    Metadata fields are flattened — keys depend on tenant's metadata_schema.
    """

    text: str = Field(description="The chunk text content.")
    score: float = Field(
        description="Cosine similarity score (0–1). Higher is more relevant."
    )
    repo_id: str = Field(description="Source repository ID.")
    source_url: str | None = Field(
        default=None, description="SharePoint or source URL."
    )
    file_name: str | None = Field(default=None, description="Source file name.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata fields (tenant-specific: domain, application etc.)",
    )

    @classmethod
    def from_dict(cls, chunk: dict) -> "ChunkResponse":
        """
        Build from the flat dict returned by VectorStoreProvider.
        Extracts known fields, puts the rest into metadata.
        """
        known = {
            "text",
            "score",
            "repo_id",
            "source_url",
            "file_name",
            "embedding",
            "tenant_id",
            "_id",
        }
        return cls(
            text=chunk.get("text", ""),
            score=chunk.get("score", 0.0),
            repo_id=str(chunk.get("repo_id", "")),
            source_url=chunk.get("source_url"),
            file_name=chunk.get("file_name"),
            metadata={k: v for k, v in chunk.items() if k not in known},
        )


class SearchResponse(BaseModel):
    """Response for POST /api/v1/{tenant_id}/search."""

    question: str
    chunks: list[ChunkResponse]
    total: int = Field(description="Number of chunks returned.")
    repos_searched: int = Field(description="Number of repos searched.")
    skipped_filters: list[str] = Field(
        default_factory=list,
        description=(
            "Filter fields that were not applied because no repo supports them. "
            "Results are returned without those filters — similarity search only."
        ),
    )


class QueryResponse(BaseModel):
    """Response for POST /api/v1/{tenant_id}/query."""

    question: str
    answer: str = Field(
        description="LLM-generated answer grounded in retrieved chunks."
    )
    chunks: list[ChunkResponse] = Field(
        description="Source chunks used to generate the answer."
    )
    total_chunks: int = Field(description="Number of chunks used as context.")


class RepoSummary(BaseModel):
    """
    Summary of a single active repo.

    filterable_fields:  all fields Atlas index supports as filters
    extractable_fields: subset of filterable_fields that LLM can
                        auto-extract from questions — content dimensions only.
                        Never includes access control fields like
                        access_group or source_id.
    """

    repo_id: str
    name: str | None = None
    source_type: str
    filterable_fields: list[str] = Field(default_factory=list)
    extractable_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Fields LLM can extract from question text. "
            "Subset of filterable_fields — never includes access_group or source_id."
        ),
    )


class ReposResponse(BaseModel):
    """Response for GET /api/v1/{tenant_id}/repos."""

    tenant_id: str
    repos: list[RepoSummary]
    total: int
