"""
VectorStoreProviderFactory — builds the correct vector store provider.

dependencies.py calls this factory — it never imports any concrete
provider class. Swapping from MongoDB Atlas to Pinecone = only change here.

The factory pattern keeps the dependency injection layer clean:
  dependencies.py
    → calls build_vectorstore_provider()
    → receives BaseVectorStoreProvider
    → passes to RetrievalService
    → RetrievalService never knows which provider is underneath

To add Pinecone or Azure AI Search:
  1. Create pinecone_provider.py inheriting BaseVectorStoreProvider
  2. Add VECTOR_STORE_PROVIDER setting to config.py
  3. Add selection logic in build_vectorstore_provider() below
  4. RetrievalService + dependencies.py require zero changes
"""

import structlog
from app.providers.vectorstore.base import BaseVectorStoreProvider

log = structlog.get_logger(__name__)


def build_vectorstore_provider() -> BaseVectorStoreProvider:
    """
    Factory — returns the configured vector store provider.

    Provider selection based on settings.VECTOR_STORE_PROVIDER:
      "mongodb_atlas" (default) → MongoDBAtlasVectorStoreProvider
      Future: "pinecone"        → PineconeVectorStoreProvider
      Future: "azure_ai_search" → AzureAISearchVectorStoreProvider

    Returns:
        BaseVectorStoreProvider — ready to call search()
    """
    from app.core.config import settings

    provider = getattr(settings, "VECTOR_STORE_PROVIDER", "mongodb_atlas")

    if provider == "mongodb_atlas":
        from app.providers.vectorstore.mongodb_atlas import (
            MongoDBAtlasVectorStoreProvider,
        )

        log.info("vectorstore_factory.building", provider="mongodb_atlas")
        return MongoDBAtlasVectorStoreProvider()

    # Future providers:
    # elif provider == "pinecone":
    #     from app.providers.vectorstore.pinecone_provider import PineconeVectorStoreProvider
    #     return PineconeVectorStoreProvider()

    raise ValueError(
        f"Unknown VECTOR_STORE_PROVIDER: '{provider}'. " f"Supported: 'mongodb_atlas'"
    )
