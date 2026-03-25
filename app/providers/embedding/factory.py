"""
EmbeddingProviderFactory — builds the correct embedding provider
from resolved tenant api_config.

RetrievalService calls this factory — it never imports any concrete
provider class. Swapping from OpenAI to Cohere = only change here.

The factory pattern keeps the service layer clean:
  RetrievalService
    → calls factory.build(api_cfg)
    → receives BaseEmbeddingProvider
    → calls embedder.embed_query(question)
    → never knows which provider is underneath
"""

import structlog
from app.providers.embedding.base import BaseEmbeddingProvider
from app.core.api_config import ResolvedApiConfig

log = structlog.get_logger(__name__)


class EmbeddingProviderFactory:
    """
    Factory for building per-request embedding providers.

    Per-request because tenants can have different:
      - API keys (genai_api_key from api_config)
      - Base URLs (Azure OpenAI endpoint)
      - Embedding models (ingestion_defaults.embedding_model)

    To add a new provider (e.g. Cohere):
      1. Create cohere_provider.py inheriting BaseEmbeddingProvider
      2. Add detection logic in build() below
      3. RetrievalService requires zero changes
    """

    def build(
        self,
        api_cfg: ResolvedApiConfig,
        api_version: str | None = None,
    ) -> BaseEmbeddingProvider:
        """
        Build the correct embedding provider for a tenant.

        Detection logic:
          - Azure OpenAI: base_url is set
          - Direct OpenAI: base_url is None
          - Future Cohere: model starts with "embed-"
          - Future local: model starts with "local/"

        Args:
            api_cfg:     Resolved tenant api_config (key, base_url, model)
            api_version: Azure OpenAI API version (from settings)

        Returns:
            BaseEmbeddingProvider ready to call embed_query()
        """
        # Future provider detection:
        # if api_cfg.embedding_model.startswith("embed-"):
        #     from app.providers.embedding.cohere_provider import CohereEmbeddingProvider
        #     return CohereEmbeddingProvider(api_key=api_cfg.api_key, ...)

        # Default: OpenAI / Azure OpenAI
        from app.providers.embedding.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            api_key=api_cfg.api_key,
            model=api_cfg.embedding_model,
            base_url=api_cfg.base_url,
            api_version=api_version,
        )
