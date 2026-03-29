"""
Reranker — re-scores retrieved chunks for relevance to the query.

Problem it solves:
  Vector similarity measures "are these texts about the same topic?"
  All SPT chunks score ~0.75 baseline due to R6a prefix.
  Irrelevant chunks (URLs, environments, CI names) rank above the
  correct answer chunk because they all share the same topic.

  Example:
    Query: "Who is primary owner of SPT?"
    Support team chunk (Service Owner Brad, Federico) → 0.796 ← ranked 1st
    Contact table chunk (Primary: Brad, Sheetal)      → 0.750 ← ranked 16th

  Reranker reads (query + chunk) TOGETHER and asks:
    "Does this chunk DIRECTLY ANSWER the question?"
    Contact table chunk → 9/10 (has Primary/Secondary distinction)
    Support team chunk  → 5/10 (has owner but no primary distinction)
    URL chunk           → 1/10 (no ownership info)

How it works:
  1. Receive top N chunks from vector search (N = top_k * 3)
  2. Send ALL chunks in ONE LLM call with relevance scoring prompt
  3. LLM returns JSON array of scores [0-10] for each chunk
  4. Re-sort by score, return top_k

One LLM call for all chunks:
  ~30 chunks × ~200 tokens = ~6000 input tokens
  Output: 30 scores = ~60 tokens
  Cost: ~$0.001 per query using tenant's existing model
  Latency: ~500-800ms additional

Uses tenant's own LLM:
  Same api_key, base_url, llm_model as LLMService
  No new API credentials needed
  Works within PwC environment constraints

Graceful degradation:
  If reranker LLM call fails → return original vector-ranked results
  Never blocks the search response
"""

import json
import asyncio
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.api_config import ResolvedApiConfig
from app.core.config import settings

log = structlog.get_logger(__name__)

# Max chars per chunk sent to reranker LLM.
# Captures R6a prefix + R6b context + first lines of content.
# Works across all chunk sizes and repo configs:
#   chunk_size=200  → sends full chunk (< 400 chars)
#   chunk_size=1024 → sends first 400 chars (prefix + R6b + content start)
#   chunk_size=2000 → sends first 400 chars (prefix + R6b summary)
#   R6b disabled    → sends prefix + first content lines
# Token cost: 30 chunks × 400 chars ≈ 3000 tokens per rerank call
MAX_RERANK_CHARS = 400

RERANK_SYSTEM_PROMPT = """\
You are a relevance scoring assistant for a document retrieval system.

Given a user question and a list of text chunks, score each chunk from 0 to 10
based on how directly and completely it answers the question.

Scoring guide:
  10 — Chunk directly and completely answers the question
   8 — Chunk contains the answer but with some extra noise
   6 — Chunk is related to the topic but doesn't directly answer
   4 — Chunk mentions some keywords but is not the answer
   2 — Chunk is about the same application but irrelevant to question
   0 — Chunk has no relevance to the question

Rules:
- Score based on ANSWER RELEVANCE, not topic similarity
- A chunk about URLs scores 0 for "who is the owner?" question
- A chunk with Primary/Secondary distinction scores higher than one without
- Return ONLY a JSON array of integer scores, one per chunk, in order
- No explanation, no other text — just the JSON array
- Example for 3 chunks: [8, 2, 0]
"""

RERANK_USER_PROMPT = """\
Question: {question}

Chunks to score:
{chunks_text}

Return a JSON array of {n} scores (0-10), one per chunk, in order.
"""


class Reranker:

    async def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int,
        api_cfg: ResolvedApiConfig,
    ) -> list[dict]:
        """
        Re-score and rerank chunks for relevance to the question.

        Args:
            question:  User's question
            chunks:    Chunks from vector search (already filtered, pre-sorted)
            top_k:     Number of chunks to return after reranking
            api_cfg:   Tenant LLM config (same model used for LLMService)

        Returns:
            Top top_k chunks sorted by reranker score descending.
            Falls back to original order if reranking fails.
        """
        if not chunks:
            return chunks

        if len(chunks) <= top_k:
            # No point reranking if we have fewer chunks than requested
            return chunks

        try:
            scores = await self._score_chunks(
                question=question,
                chunks=chunks,
                api_cfg=api_cfg,
            )

            # Attach reranker scores and sort
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = score

            reranked = sorted(
                chunks,
                key=lambda c: c.get("rerank_score", 0),
                reverse=True,
            )

            log.debug(
                "reranker.complete",
                question_len=len(question),
                candidates=len(chunks),
                returned=top_k,
                top_score=reranked[0].get("rerank_score") if reranked else 0,
            )

            return reranked[:top_k]

        except Exception as e:
            # Graceful degradation — return original vector-ranked results
            log.warning(
                "reranker.failed_graceful_degradation",
                error=str(e),
                returning_original=top_k,
            )
            return chunks[:top_k]

    async def _score_chunks(
        self,
        question: str,
        chunks: list[dict],
        api_cfg: ResolvedApiConfig,
    ) -> list[int]:
        """
        Call LLM to score all chunks in a single request.

        Returns list of integer scores 0-10, one per chunk.
        Falls back to uniform scores on parse failure.
        """
        # Build numbered chunk list for LLM
        # Send only first MAX_RERANK_CHARS per chunk — captures R6a prefix
        # + R6b context + first content lines across all chunk sizes
        # without bloating the context window
        chunks_text = "\n\n".join(
            [
                f"Chunk {i + 1}:\n{chunk.get('text', '')[:MAX_RERANK_CHARS]}"
                for i, chunk in enumerate(chunks)
            ]
        )

        prompt = RERANK_USER_PROMPT.format(
            question=question,
            chunks_text=chunks_text,
            n=len(chunks),
        )

        llm_kwargs: dict = dict(
            model=api_cfg.llm_model,
            api_key=api_cfg.api_key,
            temperature=0,
            max_tokens=200,
        )
        if api_cfg.base_url:
            llm_kwargs["base_url"] = api_cfg.base_url
        if settings.OPENAI_API_VERSION:
            llm_kwargs["openai_api_version"] = settings.OPENAI_API_VERSION

        llm = ChatOpenAI(**llm_kwargs)
        messages = [
            SystemMessage(content=RERANK_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Parse JSON array of scores
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)

        if not isinstance(scores, list):
            raise ValueError(f"Expected list, got {type(scores)}")

        # Pad or truncate to match chunk count
        scores = scores[: len(chunks)]
        while len(scores) < len(chunks):
            scores.append(0)

        return [int(s) for s in scores]
