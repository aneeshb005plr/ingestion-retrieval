"""
Search request filters schema.

SearchFilters is intentionally flat and simple from the caller's perspective.
The caller passes access control dimensions as field → list of allowed values.
LLM extraction is completely internal — never exposed in the API.

API request example:
  {
    "question": "Who is the owner of Smart Pricing Tool?",
    "filters": {
      "application": ["Smart Pricing Tool", "LeaveApp"],
      "access_group": ["xlos_all", "all"]
    }
  }

Multiple fields supported — caller adds any filterable field:
  Single field:
    { "application": ["SPT"] }

  Multiple fields (ANDed):
    { "application": ["SPT", "LeaveApp"], "domain": ["XLOS", "HR"] }

  App + document restriction:
    { "application": ["SPT"], "access_group": ["xlos_all", "all"] }

  Source-level access:
    { "source_id": ["01ABC...", "01DEF..."] }

  No filters (search all):
    {}

Semantics:
  Multiple fields  → ANDed  (must satisfy ALL dimensions)
  Multiple values  → ORed   (must match ANY value in the list)
  include_general  → adds general/shared docs to each dimension

Internal processing (never in API):
  FilterExtractor may add single-value hints from the question
  for fields NOT already provided by the caller.
  These are merged before building the NormalisedFilter.
  Caller-provided values always take precedence.
"""

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """
    Flat access control filters from the API caller.

    filters: dict[str, list[str]]
        Each key is a filterable field name (must be in repo's filterable_fields).
        Each value is a list of allowed values for that field.
        Multiple fields are ANDed. Values within a field are ORed.

        Examples:
          { "application": ["SPT", "LeaveApp"] }
          → (application=SPT OR application=LeaveApp OR is_general=true)

          { "application": ["SPT"], "access_group": ["xlos_all", "all"] }
          → (application=SPT OR is_general=true)
            AND (access_group=xlos_all OR access_group=all OR is_general=true)

    include_general:
        If True and repo has general_flag_field, general/shared docs
        are included in each dimension's OR clause.
        Default True — general docs are always visible.
    """

    filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Access control filters — field → list of allowed values. "
            "Multiple fields are ANDed. Values within a field are ORed. "
            "Examples: "
            "{application: ['SPT', 'LeaveApp']} or "
            "{application: ['SPT'], access_group: ['xlos_all', 'all']} or "
            "{source_id: ['01ABC...', '01DEF...']}"
        ),
    )
    include_general: bool = Field(
        default=True,
        description=(
            "If True and repo has a general_flag_field, "
            "general/shared docs are included alongside filtered ones."
        ),
    )
