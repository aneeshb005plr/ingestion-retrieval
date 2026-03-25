"""
BaseEmbeddingProvider — abstract interface for query embedding providers.

Retrieval-specific embedding interface:
  - Only embed_query() needed (single question, not batch documents)
  - Different from ingestion worker's BaseEmbeddingProvider which batches documents

Implementing a new provider (e.g. Azure OpenAI, Cohere, local model):
  1. Create app/providers/embedding/cohere_provider.py
  2. Inherit from BaseEmbeddingProvider
  3. Implement embed_query() and the properties
  4. Update dependencies.py factory to select it from tenant api_config

The key insight: RetrievalService never imports OpenAIEmbeddingProvider directly.
It only calls embedder.embed_query(question).
Swapping the provider = zero changes to RetrievalService.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for similarity search.

        Must use the SAME model that was used during document ingestion.
        Different model = incompatible vector space = garbage results.

        Args:
            text: The user's question or search query

        Returns:
            List of floats (embedding vector)

        Raises:
            EmbeddingError — if the provider call fails
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model. e.g. 'text-embedding-3-small'"""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vector. e.g. 1536"""
        pass
