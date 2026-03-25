"""
OpenAI Embedding Provider for the Retrieval Service.

Uses LangChain's OpenAIEmbeddings under the hood — same library
as the ingestion worker, ensuring vector space compatibility.

Supports:
  - Direct OpenAI endpoint (api.openai.com)
  - Azure OpenAI endpoint (via base_url + api_version)
  - Per-tenant API keys (from tenant.api_config.genai_api_key)

Switching to a different embedding provider:
  1. Create a new file (e.g. cohere_provider.py)
  2. Inherit BaseEmbeddingProvider, implement embed_query()
  3. Update build_embedding_provider() factory in this package
"""

import structlog
from langchain_openai import OpenAIEmbeddings

from app.providers.embedding.base import BaseEmbeddingProvider
from app.core.exceptions import EmbeddingError

log = structlog.get_logger(__name__)

# Dimensions per model — used for validation and metadata
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        api_version: str | None = None,
    ) -> None:
        """
        Args:
            api_key:     OpenAI or Azure OpenAI API key
            model:       Embedding model name — must match ingestion model
            base_url:    Azure OpenAI base URL (None = use OpenAI default)
            api_version: Azure OpenAI API version (e.g. 2024-08-01-preview)
        """
        kwargs: dict = dict(
            api_key=api_key,
            model=model,
        )
        if base_url:
            kwargs["base_url"] = base_url
        if api_version:
            kwargs["openai_api_version"] = api_version

        self._embeddings = OpenAIEmbeddings(**kwargs)
        self._model = model
        self._dims = _MODEL_DIMENSIONS.get(model, 1536)

        log.debug(
            "embedding_provider.initialised",
            model=model,
            dims=self._dims,
            azure=base_url is not None,
        )

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        Uses LangChain's native aembed_query() — truly async, no thread pool.
        """
        try:
            vector = await self._embeddings.aembed_query(text)
            log.debug(
                "embedding_provider.embedded",
                model=self._model,
                dims=len(vector),
            )
            return vector
        except Exception as e:
            log.error(
                "embedding_provider.failed",
                model=self._model,
                error=str(e),
            )
            raise EmbeddingError(
                f"Failed to embed query with model '{self._model}': {e}"
            ) from e

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims


def build_embedding_provider(
    api_key: str,
    model: str,
    base_url: str | None = None,
    api_version: str | None = None,
) -> BaseEmbeddingProvider:
    """
    Factory function — returns the correct embedding provider.

    Currently only OpenAI is supported. To add Cohere or a local model:
      1. Add provider class in its own file
      2. Add selection logic here based on model name or config

    Called by RetrievalService with tenant's resolved api_config.
    """
    # Future: detect provider from model name or explicit provider field
    # e.g. if model.startswith("embed-"): return CohereEmbeddingProvider(...)
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        api_version=api_version,
    )
