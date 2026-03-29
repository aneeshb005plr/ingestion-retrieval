"""
MongoDB Atlas Vector Search Provider.

Implements BaseVectorStoreProvider using direct $vectorSearch aggregation
pipeline on pymongo collection — no LangChain internals needed.

Filter translation:
  NormalisedFilter → MongoDB $vectorSearch pre_filter dict

  FieldCondition (single value):
    → { field: value }

  FieldCondition (multiple values):
    → { "$or": [{ field: v1 }, { field: v2 }] }

  FilterConditionWithGeneral (has general flag):
    → { "$or": [{ field: v1 }, { general_field: general_value }] }

  Multiple FieldConditions in must:
    → { "$and": [ clause1, clause2, ... ] }

  tenant_id + repo_id always first in must — security boundary.

Index naming:
  Default: vidx_repo_{repo_id}
  One index per repo — matches that repo's filterable_fields exactly.
  Prevents silent filter bypass when repos have different field sets.

  Override: repo.vector_config.index_name for custom names.

numCandidates:
  max(top_k * 20, 150) — ensures full collection coverage for small repos.
  Atlas HNSW traverses numCandidates docs before ranking.
  If numCandidates < collection size, relevant chunks may be missed.

Swap guide:
  Replace with AzureAISearchProvider, PineconeProvider etc.
  by implementing _translate_filter() + search() for that provider.
  Zero changes to FilterBuilder or RetrievalService.
"""

import asyncio
import structlog
from typing import Any
from pymongo import MongoClient

from app.providers.vectorstore.base import BaseVectorStoreProvider, SearchResult
from app.providers.vectorstore.filters import (
    NormalisedFilter,
    FieldCondition,
    FilterConditionWithGeneral,
)
from app.core.config import settings
from app.core.exceptions import VectorSearchError

log = structlog.get_logger(__name__)

VECTOR_COLLECTION = "vector_store"
TEXT_KEY = "text"
EMBEDDING_KEY = "embedding"


class MongoDBAtlasVectorStoreProvider(BaseVectorStoreProvider):

    def __init__(self) -> None:
        self._sync_client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )
        self._collection = self._sync_client[settings.MONGO_DB_NAME][VECTOR_COLLECTION]
        log.info("vectorstore_provider.initialized", type="mongodb_atlas")

    # ── Filter translation ────────────────────────────────────────────────────

    def _translate_filter(self, f: NormalisedFilter) -> dict:
        """
        Translate NormalisedFilter → MongoDB $vectorSearch pre_filter dict.

        Single condition, single value:
          → { field: value }

        Single condition, multiple values:
          → { "$or": [{ field: v1 }, { field: v2 }] }

        Condition with general flag:
          → { "$or": [{ field: v1 }, { general_field: general_value }] }

        Multiple conditions:
          → { "$and": [ clause1, clause2, ... ] }
        """
        if f.is_empty:
            return {}

        clauses = []
        for condition in f.must:
            clause = self._translate_condition(condition)
            clauses.append(clause)

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    def _translate_condition(self, condition: FieldCondition) -> dict:
        """Translate one FieldCondition to a MongoDB clause."""

        # Build OR alternatives for this field's values
        alternatives = [{condition.field: v} for v in condition.values]

        # Add general flag alternative if present
        if isinstance(condition, FilterConditionWithGeneral) and condition.has_general:
            general_alt = {condition.general_field: condition.general_value}
            if general_alt not in alternatives:
                alternatives.append(general_alt)

        if len(alternatives) == 1:
            return alternatives[0]

        return {"$or": alternatives}

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _build_pipeline(
        self,
        question_vector: list[float],
        index_name: str,
        pre_filter: dict,
        top_k: int,
    ) -> list[dict]:
        """
        Build $vectorSearch aggregation pipeline.

        numCandidates = max(top_k * 20, 150)
          → ensures Atlas HNSW traverses enough candidates
          → for small collections (< 150 docs): covers everything
          → for larger collections: 20x oversampling gives good recall
        """
        num_candidates = max(top_k * 20, 150)

        vector_search_stage: dict = {
            "$vectorSearch": {
                "index": index_name,
                "path": EMBEDDING_KEY,
                "queryVector": question_vector,
                "numCandidates": num_candidates,
                "limit": top_k,
            }
        }

        if pre_filter:
            vector_search_stage["$vectorSearch"]["filter"] = pre_filter

        return [
            vector_search_stage,
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {EMBEDDING_KEY: 0}},
        ]

    # ── Search ────────────────────────────────────────────────────────────────

    def _execute_search(
        self,
        question_vector: list[float],
        index_name: str,
        pre_filter: dict,
        top_k: int,
    ) -> list[SearchResult]:
        pipeline = self._build_pipeline(
            question_vector=question_vector,
            index_name=index_name,
            pre_filter=pre_filter,
            top_k=top_k,
        )

        results = []
        for doc in self._collection.aggregate(pipeline):
            text = doc.pop(TEXT_KEY, "")
            score = doc.pop("score", 0.0)
            repo_id = str(doc.pop("repo_id", ""))
            doc.pop("_id", None)
            doc.pop("tenant_id", None)
            doc.pop(EMBEDDING_KEY, None)
            results.append(
                SearchResult(
                    text=text,
                    score=score,
                    repo_id=repo_id,
                    metadata=doc,
                )
            )

        return results

    def _build_hybrid_pipeline(
        self,
        question: str,
        question_vector: list[float],
        vector_index_name: str,
        search_index_name: str,
        pre_filter: dict,
        top_k: int,
        alpha: float,
    ) -> list[dict]:
        """
        Build hybrid search pipeline using $rankFusion.

        Combines:
          $vectorSearch  — semantic similarity (embedding-based)
          $search        — BM25 full-text keyword matching

        IMPORTANT — $rankFusion sub-pipeline rules (per Atlas docs):
          Sub-pipelines are "selection pipelines" — they CANNOT contain
          stages that modify documents ($set, $project, $addFields etc).
          Only retrieval/ordering stages allowed: $vectorSearch, $search,
          $match, $sort, $limit, $sample, $geoNear.
          All document modifications must happen AFTER $rankFusion.
        """
        num_candidates = max(top_k * 20, 150)

        # Vector search sub-pipeline — NO $set or $project inside
        vector_pipeline: list[dict] = [
            {
                "$vectorSearch": {
                    "index": vector_index_name,
                    "path": EMBEDDING_KEY,
                    "queryVector": question_vector,
                    "numCandidates": num_candidates,
                    "limit": top_k,
                    **({"filter": pre_filter} if pre_filter else {}),
                }
            },
        ]

        # BM25 full-text search sub-pipeline — NO $set or $project inside
        bm25_pipeline: list[dict] = [
            {
                "$search": {
                    "index": search_index_name,
                    "text": {
                        "query": question,
                        "path": TEXT_KEY,
                    },
                }
            },
            {"$limit": top_k},
        ]

        # Full pipeline:
        # 1. $rankFusion combines sub-pipelines (no modifications inside)
        # 2. $limit after rankFusion
        # 3. $set score from rankFusionScore metadata (AFTER $rankFusion)
        # 4. $project to remove embedding (AFTER $rankFusion)
        return [
            {
                "$rankFusion": {
                    "input": {
                        "pipelines": {
                            "vector": vector_pipeline,
                            "text": bm25_pipeline,
                        }
                    },
                    "combination": {
                        "weights": {
                            "vector": alpha,
                            "text": round(1.0 - alpha, 4),
                        }
                    },
                }
            },
            {"$limit": top_k},
            {"$addFields": {"score": {"$meta": "score"}}},
            {"$project": {EMBEDDING_KEY: 0}},
        ]

    def _execute_hybrid_search(
        self,
        question: str,
        question_vector: list[float],
        vector_index_name: str,
        search_index_name: str,
        pre_filter: dict,
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        pipeline = self._build_hybrid_pipeline(
            question=question,
            question_vector=question_vector,
            vector_index_name=vector_index_name,
            search_index_name=search_index_name,
            pre_filter=pre_filter,
            top_k=top_k,
            alpha=alpha,
        )
        results = []
        for doc in self._collection.aggregate(pipeline):
            text = doc.pop(TEXT_KEY, "")
            score = doc.pop("score", 0.0)
            repo_id = str(doc.pop("repo_id", ""))
            doc.pop("_id", None)
            doc.pop("tenant_id", None)
            doc.pop(EMBEDDING_KEY, None)
            results.append(
                SearchResult(
                    text=text,
                    score=score,
                    repo_id=repo_id,
                    metadata=doc,
                )
            )
        return results

    async def search(
        self,
        question_vector: list[float],
        normalised_filter: NormalisedFilter,
        index_name: str,
        top_k: int,
        question: str = "",
        hybrid_search_enabled: bool = False,
        hybrid_alpha: float = 0.7,
        search_index_name: str = "",
        **kwargs: Any,
    ) -> list[SearchResult]:
        """
        Execute vector search or hybrid search depending on config.

        Pure vector:  hybrid_search_enabled=False (default)
        Hybrid:       hybrid_search_enabled=True
                      Requires search_index_name and question.
        """
        pre_filter = self._translate_filter(normalised_filter)

        try:
            if hybrid_search_enabled and question and search_index_name:
                results = await asyncio.to_thread(
                    self._execute_hybrid_search,
                    question,
                    question_vector,
                    index_name,
                    search_index_name,
                    pre_filter,
                    top_k,
                    hybrid_alpha,
                )
                log.debug(
                    "vectorstore.hybrid_searched",
                    vector_index=index_name,
                    search_index=search_index_name,
                    results=len(results),
                    top_k=top_k,
                    alpha=hybrid_alpha,
                )
            else:
                results = await asyncio.to_thread(
                    self._execute_search,
                    question_vector,
                    index_name,
                    pre_filter,
                    top_k,
                )
                log.debug(
                    "vectorstore.searched",
                    index=index_name,
                    results=len(results),
                    top_k=top_k,
                    filter=normalised_filter.describe(),
                )
            return results

        except Exception as e:
            log.error(
                "vectorstore.search_failed",
                index=index_name,
                error=str(e),
            )
            raise VectorSearchError(
                f"MongoDB Atlas search failed on index '{index_name}': {e}"
            ) from e

    def close(self) -> None:
        self._sync_client.close()
        log.info("vectorstore_provider.closed")
