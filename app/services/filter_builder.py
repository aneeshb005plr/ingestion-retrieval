"""
FilterBuilder — builds NormalisedFilter from SearchFilters + repo config.

Three-layer filter model:
  must_filters   → always applied (hard access boundary)
  should_filters → resolved value (from LLM extraction or full scope fallback)
  extracted      → LLM-narrowed value from should_filters scope

Priority:
  must_filters field   → always applied as-is
  extracted field      → narrows should_filters scope (validated within scope)
  should_filters field → full scope fallback if nothing extracted

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
            repo:               Repo document from MongoDB
            filters:            SearchFilters with must/should/auto_extract
            extracted_metadata: LLM-narrowed values from should_filters scope
                                Already validated within should_filters scope.

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

        # ── Resolve effective filters ─────────────────────────────────────────
        # Priority per field:
        #   1. must_filters  → hard boundary, always applied
        #   2. extracted     → LLM narrowed within should_filters scope
        #   3. should_filters → full scope fallback if nothing extracted

        effective: dict[str, list[str]] = {}

        # Step 1 — must_filters (hard, always applied)
        for field, values in filters.must_filters.items():
            if field not in filterable:
                log.debug(
                    "filter_builder.skipped_unsupported",
                    repo_id=repo["_id"],
                    field=field,
                    layer="must_filters",
                    reason="not in filterable_fields",
                )
                continue
            clean = [v for v in values if v]
            if clean:
                effective[field] = clean

        # Step 2 — should_filters resolved by extracted or full scope
        for field, allowed_values in filters.should_filters.items():
            if field not in filterable:
                log.debug(
                    "filter_builder.skipped_unsupported",
                    repo_id=repo["_id"],
                    field=field,
                    layer="should_filters",
                    reason="not in filterable_fields",
                )
                continue

            if field in effective:
                # Already set by must_filters — never override
                continue

            if extracted_metadata and field in extracted_metadata:
                # LLM narrowed to specific value within allowed scope
                value = extracted_metadata[field]
                if value in allowed_values:
                    effective[field] = [value]  # narrowed
                else:
                    # Extracted value outside scope — use full scope
                    log.debug(
                        "filter_builder.extraction_out_of_scope",
                        repo_id=repo["_id"],
                        field=field,
                        extracted=value,
                    )
                    effective[field] = [v for v in allowed_values if v]
            else:
                # No extraction — use full scope
                effective[field] = [v for v in allowed_values if v]

        if not effective:
            return NormalisedFilter(must=conditions)

        # ── Build FieldCondition per dimension ────────────────────────────────
        for field_name, values in effective.items():
            use_general = (
                filters.include_general
                and general_field
                and general_field in filterable
                and field_name != general_field
            )
            conditions.append(
                FilterConditionWithGeneral(
                    field=field_name,
                    values=values,
                    general_field=general_field if use_general else None,
                    general_value=general_value if use_general else None,
                )
            )

        return NormalisedFilter(must=conditions)

    def describe(self, f: NormalisedFilter) -> str:
        return f.describe()
