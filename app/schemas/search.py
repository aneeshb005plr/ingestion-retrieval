"""
Search filter schema — must_filters, should_filters, auto_extract.

Three-layer filter model:

  must_filters (hard boundary):
    Always applied — no exceptions.
    Consumer resolves user permissions and sends here.
    Platform never overrides.
    e.g. access_key: ["SPT::general", "FF::general", "FF::restricted"]

  should_filters (soft scope):
    User's allowed scope — platform narrows using LLM if auto_extract=True.
    Falls back to full list if question is vague or auto_extract=False.
    e.g. application: ["SPT", "FF", "LeaveApp", ...30 apps]

  auto_extract (boolean):
    Should platform try to narrow should_filters from question?
    true  → LLM extracts specific value from question, narrows should_filters
    false → should_filters applied as-is (full scope)

Why this design:
  Platform is generic — does not know what access_group, access_key,
  schema_name or any other field means.
  Consumer owns their access model and resolves it before calling.
  Platform applies whatever consumer sends, blindly and correctly.

  Works for ANY consumer:
    DocAssist:   must={access_key:[...]}, should={application:[30 apps]}
    SmartQuery:  must={schema_name:[...]}, should={table_name:[20 tables]}
    Simple tool: must={}, should={}  → no restrictions
    Admin:       must={}, should={}  → sees everything in tenant scope
"""

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """
    New filter model — must_filters + should_filters + auto_extract.

    Replaces old SearchFilters with flat filters dict.
    Backward compatible — old 'filters' field mapped to must_filters.

    must_filters:
      Hard access boundaries — always applied.
      field → list[str] (OR within field, AND across fields)
      e.g. {
        "access_key": ["SPT::general", "FF::general", "FF::restricted"],
        "access_group": ["general"]
      }

    should_filters:
      Soft scope — LLM narrows if auto_extract=True and question is specific.
      Falls back to full list if nothing detected.
      e.g. {
        "application": ["SPT", "FF", "LeaveApp", ... 30 apps]
      }

    auto_extract:
      True  → LLM tries to narrow should_filters from question
      False → should_filters applied as-is
    """

    must_filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Hard access boundaries — always applied, never overridden. "
            "Consumer resolves user permissions and sends here. "
            "field → list of allowed values (OR within, AND across). "
            "e.g. {access_key: ['SPT::general', 'FF::restricted']}"
        ),
    )
    should_filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Soft scope — user's allowed values for content dimensions. "
            "Platform narrows using LLM if auto_extract=True and question is specific. "
            "Falls back to full list if question is vague. "
            "e.g. {application: ['SPT', 'FF', 'LeaveApp', ...30 apps]}"
        ),
    )
    auto_extract: bool = Field(
        default=True,
        description=(
            "If True, platform uses LLM to narrow should_filters from question. "
            "If False, should_filters applied as-is (full scope). "
            "Set False when consumer already sent specific value in must_filters."
        ),
    )

    # ── Backward compatibility ────────────────────────────────────────────────
    # Old API sent: filters: { filters: {...}, include_general: true }
    # New API uses must_filters + should_filters
    # Keep include_general for repos that still use general_flag_field
    include_general: bool = Field(
        default=True,
        description=(
            "If True and repo has general_flag_field, "
            "general/shared docs included in each dimension. "
            "Kept for backward compatibility."
        ),
    )
