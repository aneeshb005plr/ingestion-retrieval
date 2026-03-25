"""
FilterExtractor — automatically extracts metadata filters from a question.

Problem it solves:
  User asks: "Who is the owner of Smart Pricing Tool?"
  Without filter → searches ALL applications → Flex Forecast owner chunk
                   scores higher → wrong answer
  With filter    → { application: "Smart Pricing Tool" } → correct answer

How it works:
  1. Gets distinct known values for each filterable field from vector_store
  2. Sends question + known values to LLM
  3. LLM identifies which value the question is about
  4. Returns filters to merge with user-provided filters

Why known values matter:
  Without them, LLM might hallucinate "Smart_Pricing_Tool" instead of
  "Smart Pricing Tool" — the exact value stored during ingestion.
  Providing the actual list forces LLM to pick a real value.
"""

import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.api_config import ResolvedApiConfig
from app.core.config import settings

log = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """You are a filter extraction assistant.
Given a user question and a list of known values for each filter field,
identify which filter values the question is specifically about.

Rules:
- Only return a filter if the question clearly refers to a specific value
- If the question is general (not about a specific application/domain), return null
- Return ONLY valid JSON, no explanation
- Use exact values from the provided lists — do not invent values

Example:
  question: "Who is the owner of Smart Pricing Tool?"
  known application values: ["Smart Pricing Tool", "Flex Forecast", "LeaveApp"]
  response: {"application": "Smart Pricing Tool"}

Example:
  question: "What are all the applications we support?"
  response: null
"""


class FilterExtractor:

    async def extract(
        self,
        question: str,
        extractable_fields: list[str],
        known_values: dict[str, list[str]],
        api_cfg: ResolvedApiConfig,
        skip_fields: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Extract content-dimension filters from question using LLM.

        Args:
            question:           User's question
            extractable_fields: Fields LLM is allowed to extract
                                (content dimensions only, never access fields)
                                e.g. ["application", "domain"]
            known_values:       Distinct values per extractable field
                                e.g. {"application": ["Smart Pricing Tool", "Flex Forecast"]}
            api_cfg:            Tenant LLM config
            skip_fields:        Fields already in access_filters — skip these

        Returns:
            dict of extracted filters e.g. {"application": "Smart Pricing Tool"}
            Empty dict if question is general or no specific value detected.
        """
        # Only extract from extractable_fields — never access control fields
        known_values = {
            k: v for k, v in known_values.items() if k in extractable_fields
        }
        # Also skip fields already provided by caller in access_filters
        if skip_fields:
            known_values = {
                k: v for k, v in known_values.items() if k not in skip_fields
            }

        if not known_values:
            return {}

        # Build context for LLM
        values_context = "\n".join(
            [
                f"{field} values: {json.dumps(vals)}"
                for field, vals in known_values.items()
                if vals
            ]
        )

        if not values_context:
            return {}

        llm_kwargs: dict = dict(
            model=api_cfg.llm_model,
            api_key=api_cfg.api_key,
            temperature=0,
            max_tokens=100,
        )
        if api_cfg.base_url:
            llm_kwargs["base_url"] = api_cfg.base_url
        if settings.OPENAI_API_VERSION:
            llm_kwargs["openai_api_version"] = settings.OPENAI_API_VERSION

        try:
            llm = ChatOpenAI(**llm_kwargs)
            messages = [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\n"
                        f"Known filter values:\n{values_context}"
                    )
                ),
            ]
            response = await llm.ainvoke(messages)
            raw = response.content.strip()

            if raw.lower() == "null" or not raw:
                return {}

            extracted = json.loads(raw)
            if not isinstance(extracted, dict):
                return {}

            # Validate — only return fields with known values
            validated = {}
            for field, value in extracted.items():
                if field in known_values and value in known_values[field]:
                    validated[field] = value

            log.debug(
                "filter_extractor.extracted",
                question=question[:50],
                extracted=validated,
            )
            return validated

        except Exception as e:
            # Extraction failure is non-fatal — fall back to no filter
            log.warning("filter_extractor.failed", error=str(e))
            return {}
