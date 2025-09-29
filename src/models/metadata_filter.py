"""
Enhanced metadata filtering models for advanced search capabilities.

Supports complex filtering operations including:
- Date range filters
- String pattern matching
- Nested field filtering
- Logical operators (AND, OR, NOT)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from pydantic import field_validator
from pydantic import ConfigDict


class FilterOperator(str, Enum):
    """Supported filter operators."""

    # Comparison operators
    EQ = "eq"          # Equal
    NEQ = "neq"        # Not equal
    GT = "gt"          # Greater than
    GTE = "gte"        # Greater than or equal
    LT = "lt"          # Less than
    LTE = "lte"        # Less than or equal

    # String operators
    CONTAINS = "contains"      # String contains
    STARTS_WITH = "starts_with"    # String starts with
    ENDS_WITH = "ends_with"      # String ends with
    REGEX = "regex"            # Regular expression match

    # Array operators
    IN = "in"              # Value in list
    NOT_IN = "not_in"      # Value not in list
    ARRAY_CONTAINS = "array_contains"  # Array contains value
    ARRAY_ANY = "array_any"      # Array contains any of values
    ARRAY_ALL = "array_all"      # Array contains all values

    # Existence operators
    EXISTS = "exists"       # Field exists
    NOT_EXISTS = "not_exists"   # Field doesn't exist

    # Date operators
    BEFORE = "before"       # Date before
    AFTER = "after"        # Date after
    BETWEEN = "between"     # Date between range


class FilterCondition(BaseModel):
    """Single filter condition."""

    field: str = Field(description="Field path to filter on (supports nested fields with dot notation)")
    operator: FilterOperator = Field(description="Filter operator to apply")
    value: Any = Field(description="Value to filter against")
    case_sensitive: bool = Field(
        default=True,
        description="Whether string comparisons are case-sensitive"
    )

    @field_validator("value", mode="before")
    def parse_dates(cls, v, info):
        """Parse date strings to datetime objects for date operators.

        Uses info.data to access sibling fields (operator) during validation.
        """
        operator = None
        try:
            operator = info.data.get("operator") if info and info.data else None
        except Exception:
            operator = None

        if operator in [FilterOperator.BEFORE, FilterOperator.AFTER]:
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v)
                except (ValueError, TypeError):
                    return v
        elif operator == FilterOperator.BETWEEN:
            if isinstance(v, list) and len(v) == 2:
                out0 = v[0]
                out1 = v[1]
                if isinstance(v[0], str):
                    try:
                        out0 = datetime.fromisoformat(v[0])
                    except (ValueError, TypeError):
                        out0 = v[0]
                if isinstance(v[1], str):
                    try:
                        out1 = datetime.fromisoformat(v[1])
                    except (ValueError, TypeError):
                        out1 = v[1]
                return [out0, out1]
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "field": "metadata.author",
                    "operator": "eq",
                    "value": "John Doe"
                },
                {
                    "field": "created_at",
                    "operator": "after",
                    "value": "2023-01-01T00:00:00"
                },
                {
                    "field": "tags",
                    "operator": "array_contains",
                    "value": "machine-learning"
                },
                {
                    "field": "title",
                    "operator": "contains",
                    "value": "neural network",
                    "case_sensitive": False
                }
            ]
        }
    )


class LogicalOperator(str, Enum):
    """Logical operators for combining filters."""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterGroup(BaseModel):
    """Group of filters with logical operator."""

    operator: LogicalOperator = Field(
        default=LogicalOperator.AND,
        description="Logical operator to combine conditions"
    )
    conditions: List[Union["FilterCondition", "FilterGroup"]] = Field(
        description="List of conditions or nested groups"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "operator": "and",
                "conditions": [
                    {
                        "field": "metadata.category",
                        "operator": "eq",
                        "value": "research"
                    },
                    {
                        "operator": "or",
                        "conditions": [
                            {
                                "field": "tags",
                                "operator": "array_contains",
                                "value": "ai"
                            },
                            {
                                "field": "tags",
                                "operator": "array_contains",
                                "value": "ml"
                            }
                        ]
                    }
                ]
            }
        }
    )


# Enable recursive model definition
FilterGroup.model_rebuild()


class MetadataFilter(BaseModel):
    """
    Enhanced metadata filter specification.

    Supports both simple dictionary format (backward compatible)
    and advanced filter groups with logical operators.
    """

    # Simple filters (backward compatible)
    simple_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Simple key-value filters (backward compatible)"
    )

    # Advanced filters
    advanced_filters: Optional[FilterGroup] = Field(
        None,
        description="Advanced filter groups with logical operators"
    )

    # Performance hints
    use_index: bool = Field(
        default=True,
        description="Whether to use metadata indexes if available"
    )
    max_candidates: Optional[int] = Field(
        None,
        description="Maximum candidates to evaluate (for performance tuning)",
        ge=1
    )

    @field_validator("advanced_filters", mode="before")
    def validate_filters(cls, v):
        """Validate filter structure before instantiation."""
        if v is not None:
            # Ensure at least one condition
            if isinstance(v, dict) and "conditions" in v:
                if not v["conditions"]:
                    raise ValueError("Filter group must have at least one condition")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "simple_filters": {
                        "category": "research",
                        "language": "en"
                    }
                },
                {
                    "advanced_filters": {
                        "operator": "and",
                        "conditions": [
                            {
                                "field": "created_at",
                                "operator": "after",
                                "value": "2023-01-01T00:00:00"
                            },
                            {
                                "field": "metadata.author",
                                "operator": "contains",
                                "value": "Smith",
                                "case_sensitive": False
                            },
                            {
                                "operator": "or",
                                "conditions": [
                                    {
                                        "field": "tags",
                                        "operator": "array_contains",
                                        "value": "important"
                                    },
                                    {
                                        "field": "priority",
                                        "operator": "gte",
                                        "value": 8
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
    )


class FilterStatistics(BaseModel):
    """Statistics about filter application."""

    total_candidates: int = Field(description="Total candidates before filtering")
    filtered_candidates: int = Field(description="Candidates after filtering")
    filter_time_ms: float = Field(description="Time spent on filtering (ms)")
    filters_applied: Dict = Field(description="Filters that were applied")
    optimization_used: Optional[str] = Field(
        None,
        description="Optimization technique used (e.g., 'index', 'cache')"
    )