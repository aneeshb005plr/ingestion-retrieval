"""
RetrievalService — orchestrates the full search pipeline.

Flow per request:
  1. Load tenant from MongoDB → resolve api_config (decrypt key)
  2. Load active repos for tenant (or scoped repo_ids)
  3. Embed the question using tenant's embedding model + api_config
  4. For each repo:
     a. Build pre_filter from repo.retrieval_config + request filters
     b. Run $vectorSearch on the tenant's Atlas index
  5. Merge results from all repos → sort by score → return top_k

Design decisions:
  - Repos searched concurrently (asyncio.gather) for performance
  - Each repo uses its own pre_filter — no cross-repo filter bleed
  - Embedding happens once — same vector reused across all repos
    (all repos for a tenant share the same index and embedding model)
  - If a repo search fails, it is logged and skipped (graceful degradation)
    so one bad repo doesn't block results from healthy repos
"""

import asyncio
import structlog
from typing import Optional

from app.core.api_config import resolve_api_config, ResolvedApiConfig
from app.core.config import settings
from app.core.exceptions import (
    TenantNotFoundError,
    TenantInactiveError,
    NoActiveReposError,
)
from app.providers.embedding.base import BaseEmbeddingProvider
from app.providers.embedding.factory import EmbeddingProviderFactory
from app.providers.vectorstore.base import BaseVectorStoreProvider
from app.repositories.tenant_repo import TenantRepository
from app.repositories.repo_repo import RepoRepository
from app.services.filter_builder import FilterBuilder
from app.services.filter_extractor import FilterExtractor
from app.schemas.search import SearchFilters

log = structlog.get_logger(__name__)


class RetrievalService:

    def __init__(
        self,
        tenant_repo: TenantRepository,
        repo_repo: RepoRepository,
        vector_store: BaseVectorStoreProvider,
        filter_builder: FilterBuilder,
        embedding_factory: EmbeddingProviderFactory,
        filter_extractor: FilterExtractor,
    ) -> None:
        self._tenant_repo = tenant_repo
        self._repo_repo = repo_repo
        self._vector_store = vector_store
        self._filter_builder = filter_builder
        self._embedding_factory = embedding_factory
        self._filter_extractor = filter_extractor

    async def search(
        self,
        question: str,
        tenant_id: str,
        filters: SearchFilters,
        top_k: int = 5,
        repo_ids: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Search for relevant document chunks matching the question.

        Args:
            question:   User's question in natural language
            tenant_id:  Tenant making the request
            filters:    Optional metadata filters (application, domain etc.)
            top_k:      Number of results to return (across all repos)
            repo_ids:   Optional list of repo IDs to scope the search

        Returns:
            Tuple of (chunks, skipped_filters).
            chunks: list of chunk dicts sorted by relevance score.
            skipped_filters: filter fields that were not supported by any repo.

        Raises:
            TenantNotFoundError  — tenant does not exist
            TenantInactiveError  — tenant is inactive
            NoActiveReposError   — tenant has no active repos
            EmbeddingError       — question embedding failed
        """
        bound_log = log.bind(tenant_id=tenant_id, top_k=top_k)
        bound_log.info("retrieval.search_start", question_len=len(question))

        # ── Step 1: Load tenant + resolve api_config ─────────────────────────
        tenant = await self._tenant_repo.get_by_id(tenant_id)
        api_cfg = await resolve_api_config(
            tenant_api_config=tenant.get("api_config"),
            tenant_ingestion_defaults=tenant.get("ingestion_defaults", {}),
        )
        bound_log.debug(
            "retrieval.api_config_resolved",
            model=api_cfg.embedding_model,
            tenant_key=api_cfg.is_tenant_key,
        )

        # ── Step 2: Load active repos ─────────────────────────────────────────
        if repo_ids:
            repos = await self._repo_repo.get_active_by_ids(
                tenant_id=tenant_id,
                repo_ids=repo_ids,
            )
        else:
            repos = await self._repo_repo.get_active_for_tenant(tenant_id)

        bound_log.debug("retrieval.repos_loaded", count=len(repos))

        # ── Step 2b: Auto-extract + merge filters ────────────────────────────
        # Always run extraction to add dimensions missing from question.
        # access_filters (caller) = access control → never overridden
        # metadata (LLM) = adds specificity to question
        # Only skips extraction if question has no extractable specifics.
        #
        # Example:
        #   caller provides: { domain: "XLOS" }  ← access restriction
        #   question: "Who owns Smart Pricing Tool?"
        #   extracted: { application: "Smart Pricing Tool" }
        #   merged: { domain: "XLOS", application: "Smart Pricing Tool" } ✅
        #
        # Example (conflict — caller wins):
        #   caller provides: { domain: "XLOS" }
        #   extracted: { domain: "Finance" }  ← LLM guessed wrong
        #   merged: { domain: "XLOS" } ← caller wins ✅
        extracted = await self._auto_extract_filters(
            question=question,
            repos=repos,
            tenant_id=tenant_id,
            api_cfg=api_cfg,
            filters=filters,
        )
        if extracted:
            bound_log.debug(
                "retrieval.filters_extracted",
                extracted=extracted,
            )

        # ── Step 3: Embed question ────────────────────────────────────────────
        embedder = self._embedding_factory.build(
            api_cfg=api_cfg,
            api_version=settings.OPENAI_API_VERSION,
        )
        question_vector = await embedder.embed_query(question)
        bound_log.debug(
            "retrieval.question_embedded",
            dims=len(question_vector),
        )

        # ── Step 4: Search each repo concurrently ─────────────────────────────
        # Fetch top_k per repo — numCandidates in the pipeline already
        # handles wide HNSW graph traversal. per_repo_k = top_k is enough
        # since we merge and re-rank across repos afterwards.
        per_repo_k = top_k

        async def search_repo(repo: dict) -> tuple[list[dict], list[str]]:
            repo_id = repo["_id"]
            index_name = repo.get("vector_config", {}).get(
                "index_name", f"vidx_repo_{repo['_id']}"
            )
            normalised_filter = self._filter_builder.build(
                tenant_id=tenant_id,
                repo=repo,
                filters=filters,
                extracted_metadata=extracted,
            )
            # Track which filter fields were skipped (not in filterable_fields)
            # Includes both caller filters and LLM extracted fields
            all_requested = set(filters.filters.keys())
            if extracted:
                all_requested.update(extracted.keys())
            filterable = set(
                repo.get("retrieval_config", {}).get("filterable_fields", [])
            )
            repo_skipped = sorted(all_requested - filterable)
            filter_desc = normalised_filter.describe()

            bound_log.debug(
                "retrieval.searching_repo",
                repo_id=repo_id,
                index=index_name,
                filter=filter_desc,
            )

            try:
                # Read hybrid search config
                # hybrid_search_enabled + hybrid_alpha → retrieval_config (generic)
                # search_index_name → vector_config (provider-specific, Atlas)
                retrieval_cfg = repo.get("retrieval_config", {})
                vector_cfg = repo.get("vector_config", {})

                hybrid_enabled = retrieval_cfg.get("hybrid_search_enabled", False)
                hybrid_alpha = retrieval_cfg.get("hybrid_alpha", 0.7)
                search_index = (
                    vector_cfg.get("search_index_name") or f"sidx_repo_{repo_id}"
                )

                results = await self._vector_store.search(
                    question_vector=question_vector,
                    normalised_filter=normalised_filter,
                    index_name=index_name,
                    top_k=per_repo_k,
                    # Hybrid search params — ignored if hybrid_search_enabled=False
                    question=question,
                    hybrid_search_enabled=hybrid_enabled,
                    hybrid_alpha=hybrid_alpha,
                    search_index_name=search_index,
                )
                # Convert SearchResult objects to dicts
                chunks = [r.to_dict() for r in results]
                bound_log.debug(
                    "retrieval.repo_results",
                    repo_id=repo_id,
                    count=len(chunks),
                )
                return chunks, repo_skipped
            except Exception as e:
                # Graceful degradation — log and skip failed repo
                bound_log.warning(
                    "retrieval.repo_search_failed",
                    repo_id=repo_id,
                    error=str(e),
                )
                return [], repo_skipped

        # Run all repo searches concurrently
        repo_results = await asyncio.gather(*[search_repo(repo) for repo in repos])

        # ── Step 5: Merge + rank + return top_k ──────────────────────────────
        all_chunks: list[dict] = []
        all_skipped: set[str] = set()
        for chunks, skipped in repo_results:
            all_chunks.extend(chunks)
            all_skipped.update(skipped)

        # Sort by score descending, return top_k
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        final = all_chunks[:top_k]

        bound_log.info(
            "retrieval.search_complete",
            total_candidates=len(all_chunks),
            returned=len(final),
            repos_searched=len(repos),
        )
        return final, sorted(all_skipped)

    async def _auto_extract_filters(
        self,
        question: str,
        repos: list[dict],
        tenant_id: str,
        api_cfg: ResolvedApiConfig,
        filters: SearchFilters,
    ) -> dict[str, str]:
        """
        Use LLM to extract content-dimension hints from the question.
        Returns dict[str, str] — single values for fields not in caller filters.
        Returns empty dict if nothing extracted or extraction fails.
        """
        # Collect extractable_fields + known values per repo
        # Each repo scoped individually — prevents cross-repo field confusion
        # e.g. repo A has extractable=[application], repo B has [table_name]
        # Without scoping: LLM sees both fields mixed — wrong
        # With scoping: each repo contributes only its own fields + values
        known_values: dict[str, list[str]] = {}

        for repo in repos:
            repo_extractable = repo.get("retrieval_config", {}).get(
                "extractable_fields", []
            )
            if not repo_extractable:
                continue

            repo_id = repo["_id"]
            for field in repo_extractable:
                values = await self._repo_repo.get_distinct_filter_values(
                    tenant_id=tenant_id,
                    field_name=field,
                    repo_ids=[repo_id],  # ← scoped to this repo only
                )
                if values:
                    # Union across repos — same field may exist in multiple repos
                    existing = known_values.get(field, [])
                    merged = list(
                        dict.fromkeys(existing + values)
                    )  # dedup, preserve order
                    known_values[field] = merged

        if not known_values:
            return {}

        # all_extractable = union of field names across repos (for extractor param)
        all_extractable = set(known_values.keys())

        # Extract filters using LLM
        # Skip fields already provided by caller in access_filters
        extracted = await self._filter_extractor.extract(
            question=question,
            extractable_fields=list(all_extractable),
            known_values=known_values,
            api_cfg=api_cfg,
            skip_fields=list(filters.filters.keys()),
        )

        if not extracted:
            return {}

        # Only return fields not already in caller's filters
        return {k: v for k, v in extracted.items() if k not in filters.filters}
