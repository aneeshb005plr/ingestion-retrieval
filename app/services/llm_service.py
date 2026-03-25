"""
LLM Service — generates answers from retrieved chunks (RAG).

Used only by POST /api/v1/{tenant_id}/query.
POST /api/v1/{tenant_id}/search bypasses this entirely — chunks only.

Uses tenant's resolved api_config:
  - genai_api_key  → api_cfg.api_key    (decrypted Fernet key)
  - genai_base_url → api_cfg.base_url   (Azure OpenAI endpoint)
  - llm_model      → api_cfg.llm_model  (e.g. gpt-4.1-mini)
"""

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.api_config import ResolvedApiConfig
from app.core.config import settings
from app.core.exceptions import LLMError

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on 
the provided document excerpts. Answer using only the information in the context.
If the context does not contain enough information to answer, say so clearly.
Do not make up information."""


class LLMService:

    async def generate_answer(
        self,
        question: str,
        chunks: list[dict],
        api_cfg: ResolvedApiConfig,
    ) -> str:
        """
        Generate an answer to the question grounded in the retrieved chunks.

        Args:
            question:  The user's question
            chunks:    Retrieved chunks from RetrievalService.search()
            api_cfg:   Resolved tenant api_config (decrypted key, model etc.)

        Returns:
            LLM-generated answer string.

        Raises:
            LLMError — if the LLM call fails.
        """
        if not chunks:
            return "I could not find any relevant documents to answer your question."

        # Build context from chunks
        context = "\n\n---\n\n".join(
            [
                f"Source: {chunk.get('file_name', 'Unknown')}\n{chunk.get('text', '')}"
                for chunk in chunks
            ]
        )

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

        try:
            llm = ChatOpenAI(**llm_kwargs)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(f"Context:\n{context}\n\n" f"Question: {question}")
                ),
            ]
            response = await llm.ainvoke(messages)
            answer = response.content

            log.debug(
                "llm_service.answer_generated",
                model=api_cfg.llm_model,
                chunks_used=len(chunks),
                answer_len=len(answer),
            )
            return answer

        except Exception as e:
            log.error(
                "llm_service.failed",
                model=api_cfg.llm_model,
                error=str(e),
            )
            raise LLMError(
                f"Failed to generate answer using model '{api_cfg.llm_model}': {e}"
            ) from e
