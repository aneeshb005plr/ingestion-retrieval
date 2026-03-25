"""
Retrieval Service configuration.
All settings read from environment variables or .env file.

Environment variables needed:
  MONGO_URI             — MongoDB Atlas connection string
  OPENAI_API_KEY        — Global fallback API key (tenant key takes priority)
  SECRET_ENCRYPTION_KEY — Fernet key for decrypting tenant genai_api_key
                          Must be the SAME key as ingestion-orchestrator + worker

Optional:
  MONGO_DB_NAME         — default: vector_platform
  LLM_MODEL             — default: gpt-4.1-mini
  EMBEDDING_MODEL       — default: text-embedding-3-small
  OPENAI_API_VERSION    — Azure OpenAI API version, default: 2024-08-01-preview
  KEY_VAULT_URL         — Azure Key Vault URL (production/AKS — overrides Fernet)
  ENVIRONMENT           — development (colored logs) | production (JSON logs)
  PORT                  — default: 8001 (local dev only, k8s uses container port)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── App ───────────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Retrieval Service"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    PORT: int = 8001

    # ── MongoDB ───────────────────────────────────────────────────────────────
    MONGO_URI: str
    MONGO_DB_NAME: str = "vector_platform"

    # ── OpenAI (global fallback — tenant api_config takes priority) ───────────
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4.1-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_VERSION: str = "2024-08-01-preview"

    # ── Provider selection ────────────────────────────────────────────────────
    # Swap vector store provider without code changes — just set env var
    VECTOR_STORE_PROVIDER: str = (
        "mongodb_atlas"  # future: "pinecone", "azure_ai_search"
    )

    # ── Encryption ────────────────────────────────────────────────────────────
    # Same key as ingestion-orchestrator and ingestion-worker
    SECRET_ENCRYPTION_KEY: str | None = None
    KEY_VAULT_URL: str | None = None  # Azure Key Vault (production/AKS)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]
