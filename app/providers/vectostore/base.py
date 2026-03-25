"""
BaseVectorStoreProvider — abstract interface for vector search providers.

Retrieval-specific interface — read-only operations only:
  - search() — similarity search with NormalisedFilter, returns scored chunks
  - _translate_filter() — provider-specific filter translation

Provider contract:
  - Accepts NormalisedFilter (provider-agnostic)
  - Translates internally to its own filter syntax
  - Returns list[SearchResult] — normalised across providers

Adding a new provider (e.g. Azure AI Search, Pinecone, Weaviate):
  1. Create app/providers/vectorstore/{name}_provider.py
  2. Inherit BaseVectorStoreProvider
  3. Implement _translate_filter(f: NormalisedFilter) -> Any
  4. Implement search()
  5. Register in VectorStoreProviderFactory

  FilterBuilder:    zero changes ✅
  RetrievalService: zero changes ✅
  SearchFilters:    zero changes ✅

Filter translation examples:
  MongoDB Atlas  → { "$and": [{ "$or": [{field: v1}, {field: v2}] }] }
  Azure AI Search→ "field eq 'v1' and (f2 eq 'v2' or f2 eq 'v3')"
  Pinecone       → { "field": { "$in": ["v1", "v2"] } }
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.providers.vectorstore.filters import NormalisedFilter


@dataclass
class SearchResult:
    """
    A single search result from a vector store.
    Normalised across providers — each provider maps its response to this.
    repo_id is a first-class field — always present, never buried in metadata.
    """

    text: str
    score: float
    repo_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flatten to dict for API response — metadata fields at top level."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "repo_id": self.repo_id,
            **self.metadata,
        }


class BaseVectorStoreProvider(ABC):

    @abstractmethod
    def _translate_filter(self, f: NormalisedFilter) -> Any:
        """
        Translate NormalisedFilter to provider-specific filter syntax.

        Called internally by search() before executing the query.
        Never called directly by RetrievalService.

        Returns:
          MongoDB Atlas:   dict  — $vectorSearch pre_filter
          Azure AI Search: str   — OData filter expression
          Pinecone:        dict  — Pinecone metadata filter
          Weaviate:        dict  — Weaviate where filter
        """
        ...

    @abstractmethod
    async def search(
        self,
        question_vector: list[float],
        normalised_filter: NormalisedFilter,
        index_name: str,
        top_k: int,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Perform similarity search and return ranked results.

        Args:
            question_vector:   Pre-computed embedding of the user's question
            normalised_filter: Provider-agnostic filter from FilterBuilder
            index_name:        Vector index / namespace to search
            top_k:             Max number of results to return
            **kwargs:          Provider-specific options

        Returns:
            List of SearchResult sorted by score descending.

        Raises:
            VectorSearchError — if the search fails
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any persistent connections (called on shutdown)."""
        ...
