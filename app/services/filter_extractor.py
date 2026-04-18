"""
FilterExtractor — automatically extracts metadata filters from a question.

Problem it solves:
  User asks: "Who is the owner of Smart Pricing Tool?"
  Without filter → searches ALL applications → Flex Forecast owner chunk
                   scores higher → wrong answer
  With filter    → { application: "Smart Pricing Tool" } → correct answer

How it works:
  1. Gets distinct known values for each filterable field from vector_store
  2. Computes abbreviations dynamically from known values
     e.g. "Smart Pricing Tool" → "SPT", "Flex Forecast" → "FF"
  3. Sends question + known values + abbreviations to LLM
  4. LLM identifies which value the question is about
  5. Returns filters to merge with user-provided filters

Why dynamic abbreviations matter:
  Without them, LLM only maps abbreviations it knows from examples.
  Generic tenants with unknown app names (e.g. "NGC", "RAD") would fail.
  Dynamic abbreviation generation works for ANY tenant, ANY application.
"""

import json
import re
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.api_config import ResolvedApiConfig
from app.core.config import settings

log = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """\
You are a filter extraction assistant.
Given a user question and a list of known values for each filter field \
(with their abbreviations), identify which filter value the question is about.

Rules:
- Return a filter if the question mentions or refers to a specific value \
OR its abbreviation
- This includes questions like "what is X", "tell me about X", "how does X work" \
— these ARE asking about X specifically, even if phrased as definitions
- If the question is truly general (no specific application/domain mentioned), return null
- Return ONLY valid JSON, no explanation
- Use exact values from the provided lists — never invent values

Example:
  question: "Who is the owner of Smart Pricing Tool?"
  application values: "Smart Pricing Tool" (SPT), "Flex Forecast" (FF)
  response: {{"application": "Smart Pricing Tool"}}

Example:
  question: "what is SPT?"
  application values: "Smart Pricing Tool" (SPT), "Flex Forecast" (FF)
  response: {{"application": "Smart Pricing Tool"}}

Example:
  question: "tell me about FF"
  application values: "Smart Pricing Tool" (SPT), "Flex Forecast" (FF)
  response: {{"application": "Flex Forecast"}}

Example:
  question: "What are all the applications we support?"
  response: null
"""


def _compute_abbreviation(value: str) -> str:
    """
    Compute abbreviation from a multi-word value.
    Takes the first letter of each significant word (uppercase).

    Examples:
      "Smart Pricing Tool"  → "SPT"
      "Flex Forecast"       → "FF"
      "Leave App"           → "LA"
      "HR Module"           → "HRM" (includes short words)
      "SingleWord"          → "" (no abbreviation for single words)
    """
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", value).split()
    if len(words) <= 1:
        return ""  # single word — no meaningful abbreviation
    return "".join(w[0].upper() for w in words if w)


def _build_values_context(known_values: dict[str, list[str]]) -> str:
    """
    Build the values context string for the LLM prompt.
    Includes dynamically computed abbreviations alongside full names.

    Example output:
      application values: "Smart Pricing Tool" (SPT), "Flex Forecast" (FF)
      domain values: "XLOS", "Finance"
    """
    lines = []
    for field, vals in known_values.items():
        if not vals:
            continue
        parts = []
        for v in vals:
            abbr = _compute_abbreviation(v)
            if abbr and abbr.lower() != v.lower():
                parts.append(f'"{v}" ({abbr})')
            else:
                parts.append(f'"{v}"')
        lines.append(f"{field} values: {', '.join(parts)}")
    return "\n".join(lines)


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
            known_values:       Distinct values per extractable field
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
        # Skip fields already provided by caller in access_filters
        if skip_fields:
            known_values = {
                k: v for k, v in known_values.items() if k not in skip_fields
            }

        if not known_values:
            return {}

        # Build context with dynamic abbreviations
        values_context = _build_values_context(known_values)

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
            log.warning("filter_extractor.failed", error=str(e))
            return {}
