"""
LLM Service — generates answers from retrieved chunks (RAG).

Used only by POST /api/v1/{tenant_id}/query.
POST /api/v1/{tenant_id}/search bypasses this entirely — chunks only.

Uses tenant's resolved api_config:
  - genai_api_key  → api_cfg.api_key    (decrypted Fernet key)
  - genai_base_url → api_cfg.base_url   (Azure OpenAI endpoint)
  - llm_model      → api_cfg.llm_model  (e.g. gpt-4.1-mini)

── Structured Output Contract ───────────────────────────────────────────────
The LLM is asked to return a JSON object — NOT free text.
This lets us know EXACTLY which chunks were used and whether an answer
was found at all.

Expected JSON from LLM:
  {
    "answer": "The owner of SPT is John Smith.",
    "answer_found": true,
    "used_chunk_indices": [0, 2]
  }

  answer             — the answer text (null if answer_found is false)
  answer_found       — false when context doesn't contain enough to answer
  used_chunk_indices — 0-based indices of chunks the LLM drew from

Fallback on parse failure:
  If the LLM returns malformed JSON, we return the raw text as the answer
  with answer_found=True and used_chunk_indices=[] (no citation rather
  than wrong citation). This is the safe production choice — the user
  still gets an answer, but no chunks are attributed.

  Logged as: llm_service.json_parse_failed (warning)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import structlog
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.api_config import ResolvedApiConfig
from app.core.config import settings
from app.core.exceptions import LLMError

log = structlog.get_logger(__name__)

# ── Structured output prompt ──────────────────────────────────────────────────
# Numbered chunks are passed so the LLM can cite by index.
# Strict JSON-only instruction — no preamble, no markdown fences.
# "used_chunk_indices" forces explicit attribution rather than implicit use.
# "answer_found: false" path prevents hallucination on out-of-scope questions.

SYSTEM_PROMPT = """\
You are a precise document assistant. Answer the question using ONLY the \
numbered document excerpts provided.

Rules:
1. Read all excerpts carefully before answering.
2. Base your answer ONLY on information present in the excerpts.
3. If the excerpts do not contain enough information to answer, set \
answer_found to false.
4. Do NOT make up information or use outside knowledge.
5. In used_chunk_indices list ONLY the excerpt numbers (0-based) that you \
actually read and used to construct your answer. Omit excerpts that were \
irrelevant.

Respond with a single JSON object and nothing else — no markdown fences, \
no preamble, no explanation outside the JSON:

{
  "answer": "<your answer, or null if answer_found is false>",
  "answer_found": <true or false>,
  "used_chunk_indices": [<0-based indices of excerpts used, empty if none>]
}"""


@dataclass
class LLMResult:
    """
    Typed result from LLMService.generate_answer().

    answer             — LLM answer text. None when answer_found is False.
    answer_found       — True if the context contained enough to answer.
    used_indices       — 0-based indices into the chunks list that the LLM
                         used. Caller filters the chunk list by these indices
                         to build cited_chunks for the API response.
    parse_failed       — True if JSON parsing failed; caller should treat
                         answer as unattributed plain text (no chunks).
    """

    answer: str | None
    answer_found: bool
    used_indices: list[int] = field(default_factory=list)
    parse_failed: bool = False


class LLMService:

    async def generate_answer(
        self,
        question: str,
        chunks: list[dict],
        api_cfg: ResolvedApiConfig,
    ) -> LLMResult:
        """
        Generate a structured answer grounded in the retrieved chunks.

        Chunks are passed to the LLM numbered (Chunk 0, Chunk 1, …).
        The LLM returns a JSON object specifying:
          - the answer text
          - whether an answer was found at all
          - which chunk indices it actually used

        Args:
            question:  The user's question.
            chunks:    Retrieved + reranked chunks from RetrievalService.search().
                       These are the candidates — the LLM may use only a subset.
            api_cfg:   Resolved tenant api_config (decrypted key, model etc.)

        Returns:
            LLMResult — structured result with answer, answer_found, used_indices.

        Raises:
            LLMError — if the LLM call itself fails (network, auth, quota).
        """
        if not chunks:
            log.info("llm_service.no_chunks", model=api_cfg.llm_model)
            return LLMResult(
                answer=None,
                answer_found=False,
                used_indices=[],
            )

        # ── Build numbered context block ──────────────────────────────────────
        # Each chunk gets a 0-based index so the LLM can cite by number.
        # file_name is included as the source label for readability.
        context_parts = []
        for idx, chunk in enumerate(chunks):
            source = chunk.get("file_name") or chunk.get("source_url") or "Unknown"
            text = chunk.get("text", "").strip()
            context_parts.append(f"[Chunk {idx}] Source: {source}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # ── Build LLM client ──────────────────────────────────────────────────
        llm_kwargs: dict = dict(
            model=api_cfg.llm_model,
            api_key=api_cfg.api_key,
            temperature=0,
            max_tokens=1000,
        )
        if api_cfg.base_url:
            llm_kwargs["base_url"] = api_cfg.base_url
        if settings.OPENAI_API_VERSION:
            llm_kwargs["openai_api_version"] = settings.OPENAI_API_VERSION

        # ── Call LLM ──────────────────────────────────────────────────────────
        try:
            llm = ChatOpenAI(**llm_kwargs)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Document excerpts:\n\n{context}"
                        f"\n\n---\n\nQuestion: {question}"
                    )
                ),
            ]
            response = await llm.ainvoke(messages)
            raw = response.content

            log.debug(
                "llm_service.raw_response",
                model=api_cfg.llm_model,
                raw_len=len(raw),
            )

        except Exception as e:
            log.error(
                "llm_service.call_failed",
                model=api_cfg.llm_model,
                error=str(e),
            )
            raise LLMError(
                f"Failed to generate answer using model '{api_cfg.llm_model}': {e}"
            ) from e

        # ── Parse structured JSON response ────────────────────────────────────
        return self._parse_llm_response(
            raw=raw,
            num_chunks=len(chunks),
            model=api_cfg.llm_model,
        )

    def _parse_llm_response(
        self,
        raw: str,
        num_chunks: int,
        model: str,
    ) -> LLMResult:
        """
        Parse the LLM's JSON response into a typed LLMResult.

        Safety rules applied after parsing:
          - used_chunk_indices clamped to valid range [0, num_chunks)
          - answer coerced to None if answer_found is False
          - Fallback on any parse error: plain text answer, no citation

        Args:
            raw:        Raw string from LLM.
            num_chunks: Length of the chunk list passed to the LLM.
                        Used to guard against hallucinated out-of-range indices.
            model:      Model name for log context only.

        Returns:
            LLMResult
        """
        # Strip markdown code fences if the LLM wrapped its JSON anyway
        # (defensive — prompt says not to, but models sometimes do)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop first line (``` or ```json) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.warning(
                "llm_service.json_parse_failed",
                model=model,
                error=str(exc),
                raw_snippet=raw[:200],
            )
            # Fallback: return the raw text as the answer, no chunk attribution.
            # Safe production choice — user gets an answer, no wrong citations.
            return LLMResult(
                answer=raw.strip() or None,
                answer_found=bool(raw.strip()),
                used_indices=[],
                parse_failed=True,
            )

        # ── Extract fields with safe defaults ─────────────────────────────────
        answer_found: bool = bool(data.get("answer_found", True))
        raw_answer = data.get("answer")
        raw_indices = data.get("used_chunk_indices", [])

        # Coerce answer to None when answer_found is False
        if not answer_found:
            answer = None
        else:
            answer = str(raw_answer).strip() if raw_answer else None
            # If LLM said answer_found=True but gave empty answer, correct it
            if not answer:
                answer_found = False

        # Guard: only keep indices that are valid integers within range
        # Prevents IndexError if LLM hallucinates an out-of-range index
        safe_indices: list[int] = []
        if isinstance(raw_indices, list):
            for idx in raw_indices:
                if isinstance(idx, int) and 0 <= idx < num_chunks:
                    safe_indices.append(idx)
                else:
                    log.warning(
                        "llm_service.invalid_chunk_index",
                        model=model,
                        index=idx,
                        num_chunks=num_chunks,
                    )

        # Deduplicate and preserve order
        seen: set[int] = set()
        deduped_indices: list[int] = []
        for idx in safe_indices:
            if idx not in seen:
                deduped_indices.append(idx)
                seen.add(idx)

        log.debug(
            "llm_service.answer_parsed",
            model=model,
            answer_found=answer_found,
            cited_chunks=len(deduped_indices),
            answer_len=len(answer) if answer else 0,
        )

        return LLMResult(
            answer=answer,
            answer_found=answer_found,
            used_indices=deduped_indices,
        )