"""
FilterBuilder — builds NormalisedFilter from SearchFilters + repo config.

Returns a provider-agnostic NormalisedFilter.
Each vector store provider translates it to its own syntax internally.

Filter construction rules:

  1. tenant_id + repo_id — always present (security boundary)

  2. access_filters (caller-provided access control):
     Each field → one FieldCondition with list of allowed values
     Multiple fields → multiple FieldConditions (ANDed by NormalisedFilter)

  3. metadata (LLM-extracted content dimensions):
     Only added for fields NOT already in access_filters
     Single value → FieldCondition with list of one

  4. include_general:
     If True and repo has general_flag_field:
       Append general flag value to each FieldCondition's values
       e.g. application=["SPT", "LeaveApp"] + is_general=true
       → FieldCondition("application", ["SPT", "LeaveApp", "true"]) ← wrong

       Actually general flag is a SEPARATE field condition appended
       to each dimension OR:
       We add it as a separate FieldCondition only if no other
       filters are present, OR we include it in each OR clause.

       Correct semantics:
         (application=SPT OR is_general=true)  ← general spans all apps
         AND
         (access_group=xlos_all OR is_general=true)  ← general spans all groups

       This means: general docs bypass ALL access dimensions
       → include general_flag in each FieldCondition's values as a
         separate field=value pair. But FieldCondition is single-field.

       Solution: for each filter FieldCondition, append the general
       flag as an additional value using a special sentinel, OR
       build a separate OR per condition that includes the general flag.

       We handle this by adding the general_flag_value to a
       SEPARATE general FieldCondition that's OR'd at the top level
       via the provider's translation — OR we keep current approach:
       general flag is added as extra value in each condition.

  The general flag approach:
    NormalisedFilter has an optional general_condition field
    that providers include as OR alongside each must condition.

    This keeps FieldCondition pure (single field)
    and lets providers handle general inclusion correctly.

Unsupported fields (not in filterable_fields) are silently skipped.
"""

import structlog
from app.schemas.search import SearchFilters
from app.providers.vectorstore.filters import (
    FieldCondition,
    NormalisedFilter,
    FilterConditionWithGeneral,
)

log = structlog.get_logger(__name__)


class FilterBuilder:

    def build(
        self,
        tenant_id: str,
        repo: dict,
        filters: SearchFilters,
        extracted_metadata: dict[str, str] | None = None,
    ) -> NormalisedFilter:
        """
        Build NormalisedFilter for a specific repo.

        Args:
            tenant_id:          Tenant making the request
            repo:               Repo document from MongoDB with retrieval_config
            filters:            Caller-provided access filters (flat dict, lists)
            extracted_metadata: LLM-extracted hints (internal, single values)
                                Only added for fields not in caller's filters.

        Returns:
            NormalisedFilter — provider-agnostic, ready for translation
        """
        retrieval_cfg = repo.get("retrieval_config", {})
        filterable = set(retrieval_cfg.get("filterable_fields", []))
        general_field = retrieval_cfg.get("general_flag_field")
        general_value = retrieval_cfg.get("general_flag_value", "true")

        # ── Always present — security boundary ───────────────────────────────
        conditions: list[FieldCondition] = [
            FieldCondition("tenant_id", [tenant_id]),
            FieldCondition("repo_id", [repo["_id"]]),
        ]

        # ── Resolve effective filter fields ───────────────────────────────────
        # access_filters: caller-provided access control (lists)
        # metadata:       LLM-extracted content dimensions (single values)
        #                 only added for fields NOT in access_filters

        effective: dict[str, list[str]] = {}

        # Access filters — must be in filterable_fields
        for field_name, values in filters.filters.items():
            if field_name not in filterable:
                log.debug(
                    "filter_builder.skipped_unsupported",
                    repo_id=repo["_id"],
                    field=field_name,
                    reason="not in filterable_fields",
                )
                continue
            clean = [v for v in values if v]
            if clean:
                effective[field_name] = clean

        # LLM extracted hints — only add fields not already in caller filters
        if extracted_metadata:
            for field_name, value in extracted_metadata.items():
                if field_name not in filterable:
                    continue
                if field_name in effective:
                    # Already provided by caller — never override
                    continue
                if value:
                    effective[field_name] = [value]

        if not effective:
            return NormalisedFilter(must=conditions)

        # ── Build FieldCondition per dimension ────────────────────────────────
        for field_name, values in effective.items():
            # Add general flag to each dimension's values
            # so general docs are included regardless of filter
            # e.g. application=["SPT", "LeaveApp"] + is_general=true
            # means: (app=SPT OR app=LeaveApp OR is_general=true)
            final_values = list(values)

            use_general = (
                filters.include_general
                and general_field
                and general_field in filterable
                and field_name != general_field  # don't add to itself
            )
            if use_general:
                # We encode general as a virtual value on the condition
                # The provider sees: field=["SPT","LeaveApp","__general__"]
                # and translates: field=SPT OR field=LeaveApp OR general_field=general_value
                # But this mixes field names — bad.
                #
                # Better: NormalisedFilter carries general_flag info separately
                # and each FieldCondition has an include_general flag.
                # Provider handles it per condition.
                pass  # handled via FieldCondition.include_general below

            conditions.append(
                FilterConditionWithGeneral(
                    field=field_name,
                    values=final_values,
                    general_field=general_field if use_general else None,
                    general_value=general_value if use_general else None,
                )
            )

        return NormalisedFilter(must=conditions)

    def describe(self, f: NormalisedFilter) -> str:
        return f.describe()
