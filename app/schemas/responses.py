"""
REST API response schemas for the Retrieval Service.

ChunkResponse   — single chunk returned from search or query
SearchResponse  — POST /api/v1/{tenant_id}/search
QueryResponse   — POST /api/v1/{tenant_id}/query
RepoResponse    — GET  /api/v1/{tenant_id}/repos

── QueryResponse change log (R7) ────────────────────────────────────────────
Before R7:
  answer: str                    ← always present, even on no-answer
  chunks: list[ChunkResponse]    ← ALL retrieved chunks, even unused ones
  total_chunks: int              ← count of ALL retrieved chunks

After R7:
  answer: str | None             ← None when answer_available is False
  answer_available: bool         ← NEW — False = answer not in documents
  chunks: list[ChunkResponse]    ← ONLY chunks the LLM cited
  cited_chunks: int              ← count of cited chunks (replaces total_chunks)

Consumers MUST check answer_available before reading answer.
When answer_available is False, chunks will always be [].
─────────────────────────────────────────────────────────────────────────────
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
        description=(
            "Additional metadata fields (tenant-specific: domain, application etc.)"
        ),
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
    """
    Response for POST /api/v1/{tenant_id}/query.

    answer_available signals whether the retrieved documents contained
    enough information to answer the question.

    When answer_available is True:
      - answer  contains the LLM-generated answer text
      - chunks  contains ONLY the document excerpts the LLM cited
      - cited_chunks > 0

    When answer_available is False:
      - answer  is None — the LLM could not answer from the context
      - chunks  is []  — no chunks are attributed (nothing to cite)
      - cited_chunks is 0

    Consumers should always check answer_available before reading answer.

    Example (answer found):
      {
        "question": "Who owns SPT?",
        "answer": "The owner of SPT is John Smith.",
        "answer_available": true,
        "chunks": [
          { "text": "...", "score": 0.91, "file_name": "SPT_Owners.pdf", ... }
        ],
        "cited_chunks": 1,
        "skipped_filters": []
      }

    Example (answer not found):
      {
        "question": "What is the GDP of France?",
        "answer": null,
        "answer_available": false,
        "chunks": [],
        "cited_chunks": 0,
        "skipped_filters": []
      }
    """

    question: str

    answer: str | None = Field(
        default=None,
        description=(
            "LLM-generated answer grounded in cited chunks. "
            "Null when answer_available is False."
        ),
    )

    answer_available: bool = Field(
        description=(
            "True if the retrieved documents contained enough information "
            "to answer the question. False = the question is out of scope "
            "for the available documents."
        ),
    )

    chunks: list[ChunkResponse] = Field(
        description=(
            "Document chunks the LLM cited when constructing the answer. "
            "Empty when answer_available is False. "
            "Will NOT include chunks that were retrieved but not used."
        ),
    )

    cited_chunks: int = Field(
        description=(
            "Number of document chunks cited by the LLM. "
            "Always equals len(chunks). Zero when answer_available is False."
        ),
    )

    skipped_filters: list[str] = Field(
        default_factory=list,
        description=(
            "Filter fields that were not applied because no repo supports them."
        ),
    )


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