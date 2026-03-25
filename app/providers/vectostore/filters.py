"""
Provider-agnostic filter representation for vector search.

Problem this solves:
  Different vector store providers use completely different filter syntaxes:
    MongoDB Atlas  → { "$and": [{ "$or": [...] }] }
    Azure AI Search → "field eq 'value' and (f2 eq 'v1' or f2 eq 'v2')"
    Pinecone       → { "field": { "$in": ["v1", "v2"] } }
    Weaviate       → { "operator": "And", "operands": [...] }

  Without this layer, FilterBuilder would produce MongoDB-specific dicts.
  Swapping providers would require changing FilterBuilder too — wrong.

Solution:
  FilterBuilder produces NormalisedFilter (provider-agnostic).
  Each provider's _translate_filter() converts to its own syntax.
  RetrievalService and FilterBuilder never know about provider syntax.

Data model:

  FieldCondition:
    field:  str         — metadata field name e.g. "application"
    values: list[str]   — allowed values, always OR'd within a field
                          single value = list of one

  NormalisedFilter:
    must: list[FieldCondition]  — all conditions AND'd together

  Semantics:
    must=[
      FieldCondition("tenant_id",   ["docassist_dev"]),
      FieldCondition("repo_id",     ["repo_abc"]),
      FieldCondition("application", ["SPT", "LeaveApp"]),
      FieldCondition("access_group",["xlos_all", "all"]),
    ]
    means:
      tenant_id = docassist_dev
      AND repo_id = repo_abc
      AND (application = SPT OR application = LeaveApp)
      AND (access_group = xlos_all OR access_group = all)

Swap guide — adding a new provider:
  1. Create new provider file e.g. azure_ai_search.py
  2. Inherit BaseVectorStoreProvider
  3. Implement _translate_filter(f: NormalisedFilter) -> str  (OData)
  4. Implement search()
  5. Register in VectorStoreProviderFactory
  6. Zero changes to FilterBuilder, RetrievalService, or any schema
"""

from dataclasses import dataclass, field


@dataclass
class FieldCondition:
    """
    A single field filter condition.

    Values are always OR'd within the field:
      FieldCondition("application", ["SPT", "LeaveApp"])
      → application = SPT OR application = LeaveApp

    Single value is just a list of one:
      FieldCondition("tenant_id", ["docassist_dev"])
      → tenant_id = docassist_dev
    """

    field: str
    values: list[str]

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("FieldCondition.field cannot be empty")
        if not self.values:
            raise ValueError(
                f"FieldCondition.values cannot be empty for field '{self.field}'"
            )
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for v in self.values:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        self.values = deduped

    @property
    def is_single(self) -> bool:
        """True if this condition has exactly one value."""
        return len(self.values) == 1

    def describe(self) -> str:
        """Human-readable description for logging."""
        if self.is_single:
            return f"{self.field}={self.values[0]}"
        values_str = " OR ".join(self.values)
        return f"({self.field}={values_str})"


@dataclass
class NormalisedFilter:
    """
    Provider-agnostic filter — all conditions AND'd together.

    Each FieldCondition OR's its values internally.
    All FieldConditions are AND'd together.

    Empty must list = no filter (search all documents).

    Example:
      NormalisedFilter(must=[
        FieldCondition("tenant_id",    ["docassist_dev"]),
        FieldCondition("repo_id",      ["repo_abc"]),
        FieldCondition("application",  ["SPT", "LeaveApp"]),
        FieldCondition("access_group", ["xlos_all", "all"]),
      ])

      Semantics:
        tenant_id = docassist_dev
        AND repo_id = repo_abc
        AND (application = SPT OR application = LeaveApp)
        AND (access_group = xlos_all OR access_group = all)
    """

    must: list[FieldCondition] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no conditions — search all documents."""
        return len(self.must) == 0

    def get_field(self, field_name: str) -> FieldCondition | None:
        """Get condition for a specific field if present."""
        for condition in self.must:
            if condition.field == field_name:
                return condition
        return None

    def describe(self) -> str:
        """Human-readable description for logging."""
        if self.is_empty:
            return "no filter"
        return " AND ".join(c.describe() for c in self.must)


class FilterConditionWithGeneral(FieldCondition):
    """
    Extended FieldCondition that carries an optional general flag.

    Used when a repo has a general_flag_field (e.g. is_general=true)
    that should be OR'd alongside the main field condition so that
    general/shared documents are always included.

    Provider translates this as:
      (field=v1 OR field=v2 OR general_field=general_value)

    When general_field is None: standard OR of values only.

    Lives in filters.py (not filter_builder.py) so providers can
    import it without creating a service → provider dependency.
    """

    def __init__(
        self,
        field: str,
        values: list[str],
        general_field: str | None = None,
        general_value: str | None = None,
    ) -> None:
        super().__init__(field=field, values=values)
        self.general_field = general_field
        self.general_value = general_value

    @property
    def has_general(self) -> bool:
        """True if this condition has a general flag to include."""
        return bool(self.general_field and self.general_value)

    def describe(self) -> str:
        parts = [f"{self.field}={v}" for v in self.values]
        if self.has_general:
            parts.append(f"{self.general_field}={self.general_value}")
        if len(parts) == 1:
            return parts[0]
        return "(" + " OR ".join(parts) + ")"
