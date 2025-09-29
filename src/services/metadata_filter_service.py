"""
Enhanced metadata filtering service with advanced filtering capabilities.

Provides sophisticated filtering operations for vector search results.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from models.metadata_filter import (
    FilterCondition,
    FilterGroup,
    FilterOperator,
    LogicalOperator,
    MetadataFilter,
    FilterStatistics,
)


class MetadataFilterService:
    """
    Service for applying enhanced metadata filters to search results.

    Supports complex filtering operations including:
    - Nested field access
    - Date range queries
    - String pattern matching
    - Logical operators (AND, OR, NOT)
    - Array operations
    """

    def __init__(self):
        """Initialize the filter service."""
        self._compiled_regexes = {}  # Cache compiled regex patterns

    def apply_filters(
        self,
        items: List[Any],
        filter_spec: Union[Dict, MetadataFilter],
        metadata_extractor=None,
    ) -> tuple[List[Any], FilterStatistics]:
        """
        Apply metadata filters to a list of items.

        Args:
            items: List of items to filter (chunks, documents, etc.)
            filter_spec: Filter specification (dict or MetadataFilter)
            metadata_extractor: Function to extract metadata from items

        Returns:
            Tuple of (filtered items, filter statistics)
        """
        start_time = datetime.now()
        total_candidates = len(items)

        # Convert dict to MetadataFilter if needed
        if isinstance(filter_spec, dict):
            metadata_filter = MetadataFilter(simple_filters=filter_spec)
        else:
            metadata_filter = filter_spec

        # Apply filters (both, in sequence)
        filtered_items = items
        if metadata_filter.simple_filters:
            filtered_items = self._apply_simple_filters(
                filtered_items, metadata_filter.simple_filters, metadata_extractor
            )
        if metadata_filter.advanced_filters:
            filtered_items = self._apply_advanced_filters(
                filtered_items, metadata_filter.advanced_filters, metadata_extractor
            )

        # Calculate statistics
        filter_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        stats = FilterStatistics(
            total_candidates=total_candidates,
            filtered_candidates=len(filtered_items),
            filter_time_ms=filter_time_ms,
            filters_applied=metadata_filter.model_dump(exclude_none=True),
            optimization_used="in-memory",
        )

        return filtered_items, stats

    def _apply_simple_filters(
        self, items: List[Any], filters: Dict, metadata_extractor=None
    ) -> List[Any]:
        """Apply simple key-value filters (backward compatible)."""
        filtered = []
        for item in items:
            metadata = self._extract_metadata(item, metadata_extractor)
            if self._matches_simple_filters(metadata, filters):
                filtered.append(item)
        return filtered

    def _matches_simple_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches simple filters."""
        for key, value in filters.items():
            if not self._check_simple_condition(metadata, key, value):
                return False
        return True

    def _check_simple_condition(
        self, metadata: Dict, key: str, filter_value: Any
    ) -> bool:
        """Check a simple filter condition."""
        # Get nested field value
        actual_value = self._get_nested_value(metadata, key)

        if actual_value is None:
            return False

        # Handle different filter value types
        if isinstance(filter_value, dict):
            # Range filters
            if "gte" in filter_value and actual_value < filter_value["gte"]:
                return False
            if "lte" in filter_value and actual_value > filter_value["lte"]:
                return False
            if "gt" in filter_value and actual_value <= filter_value["gt"]:
                return False
            if "lt" in filter_value and actual_value >= filter_value["lt"]:
                return False
            if "contains" in filter_value:
                if not self._contains_check(actual_value, filter_value["contains"]):
                    return False
        elif isinstance(filter_value, list):
            # Value must be in list
            if actual_value not in filter_value:
                return False
        else:
            # Exact match
            if actual_value != filter_value:
                return False

        return True

    def _apply_advanced_filters(
        self, items: List[Any], filter_group: FilterGroup, metadata_extractor=None
    ) -> List[Any]:
        """Apply advanced filter groups with logical operators."""
        filtered = []
        for item in items:
            metadata = self._extract_metadata(item, metadata_extractor)
            if self._evaluate_filter_group(metadata, filter_group):
                filtered.append(item)
        return filtered

    def _evaluate_filter_group(self, metadata: Dict, group: FilterGroup) -> bool:
        """Evaluate a filter group against metadata."""
        if group.operator == LogicalOperator.AND:
            return all(
                self._evaluate_condition(metadata, cond) for cond in group.conditions
            )
        elif group.operator == LogicalOperator.OR:
            return any(
                self._evaluate_condition(metadata, cond) for cond in group.conditions
            )
        elif group.operator == LogicalOperator.NOT:
            return not any(
                self._evaluate_condition(metadata, cond) for cond in group.conditions
            )
        else:
            raise ValueError(f"Unknown logical operator: {group.operator}")

    def _evaluate_condition(
        self, metadata: Dict, condition: Union[FilterCondition, FilterGroup]
    ) -> bool:
        """Evaluate a single condition or nested group."""
        if isinstance(condition, FilterGroup):
            return self._evaluate_filter_group(metadata, condition)
        else:
            return self._evaluate_filter_condition(metadata, condition)

    def _evaluate_filter_condition(
        self, metadata: Dict, condition: FilterCondition
    ) -> bool:
        """Evaluate a single filter condition."""
        value = self._get_nested_value(metadata, condition.field)
        filter_value = condition.value
        operator = condition.operator

        # Existence checks
        if operator == FilterOperator.EXISTS:
            return value is not None
        elif operator == FilterOperator.NOT_EXISTS:
            return value is None

        # If value doesn't exist and we need it, return False
        if value is None:
            return False

        # Comparison operators
        if operator == FilterOperator.EQ:
            return value == filter_value
        elif operator == FilterOperator.NEQ:
            return value != filter_value
        elif operator == FilterOperator.GT:
            return value > filter_value
        elif operator == FilterOperator.GTE:
            return value >= filter_value
        elif operator == FilterOperator.LT:
            return value < filter_value
        elif operator == FilterOperator.LTE:
            return value <= filter_value

        # String operators
        elif operator == FilterOperator.CONTAINS:
            return self._string_contains(value, filter_value, condition.case_sensitive)
        elif operator == FilterOperator.STARTS_WITH:
            return self._string_starts_with(
                value, filter_value, condition.case_sensitive
            )
        elif operator == FilterOperator.ENDS_WITH:
            return self._string_ends_with(
                value, filter_value, condition.case_sensitive
            )
        elif operator == FilterOperator.REGEX:
            return self._regex_match(value, filter_value, condition.case_sensitive)

        # Array operators
        elif operator == FilterOperator.IN:
            return isinstance(filter_value, (list, tuple, set)) and value in filter_value
        elif operator == FilterOperator.NOT_IN:
            return isinstance(filter_value, (list, tuple, set)) and value not in filter_value
        elif operator == FilterOperator.ARRAY_CONTAINS:
            return self._array_contains(value, filter_value)
        elif operator == FilterOperator.ARRAY_ANY:
            return self._array_contains_any(value, filter_value)
        elif operator == FilterOperator.ARRAY_ALL:
            return self._array_contains_all(value, filter_value)

        # Date operators
        elif operator == FilterOperator.BEFORE:
            return self._date_before(value, filter_value)
        elif operator == FilterOperator.AFTER:
            return self._date_after(value, filter_value)
        elif operator == FilterOperator.BETWEEN:
            return self._date_between(value, filter_value)

        else:
            raise ValueError(f"Unknown filter operator: {operator}")

    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split(".")
        value = obj

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None

        return value

    def _extract_metadata(self, item: Any, extractor=None) -> Dict:
        """Extract metadata from an item."""
        if extractor:
            return extractor(item)

        # Default extraction strategies
        if hasattr(item, "metadata"):
            metadata = item.metadata
            if hasattr(metadata, "__dict__"):
                return metadata.__dict__
            elif isinstance(metadata, dict):
                return metadata
            else:
                return {}
        elif isinstance(item, dict):
            return item.get("metadata", {})
        else:
            return {}

    # String operation helpers
    def _string_contains(self, value: str, pattern: str, case_sensitive: bool) -> bool:
        """Check if string contains pattern."""
        if not isinstance(value, str):
            return False
        if not case_sensitive:
            return pattern.lower() in value.lower()
        return pattern in value

    def _string_starts_with(
        self, value: str, pattern: str, case_sensitive: bool
    ) -> bool:
        """Check if string starts with pattern."""
        if not isinstance(value, str):
            return False
        if not case_sensitive:
            return value.lower().startswith(pattern.lower())
        return value.startswith(pattern)

    def _string_ends_with(
        self, value: str, pattern: str, case_sensitive: bool
    ) -> bool:
        """Check if string ends with pattern."""
        if not isinstance(value, str):
            return False
        if not case_sensitive:
            return value.lower().endswith(pattern.lower())
        return value.endswith(pattern)

    def _regex_match(self, value: str, pattern: str, case_sensitive: bool) -> bool:
        """Check if string matches regex pattern."""
        if not isinstance(value, str):
            return False

        # Cache compiled regex
        cache_key = f"{pattern}_{case_sensitive}"
        try:
            compiled = self._compiled_regexes.get(cache_key)
            if compiled is None:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern, flags)
                self._compiled_regexes[cache_key] = compiled
            return bool(compiled.search(value))
        except re.error:
            return False

    def _contains_check(self, value: Any, pattern: Any) -> bool:
        """Generic contains check for strings and lists."""
        if isinstance(value, str) and isinstance(pattern, str):
            return pattern in value
        elif isinstance(value, list):
            return pattern in value
        return False

    # Array operation helpers
    def _array_contains(self, value: Any, item: Any) -> bool:
        """Check if array contains item."""
        if not isinstance(value, list):
            return False
        return item in value

    def _array_contains_any(self, value: Any, items: List) -> bool:
        """Check if array contains any of the items."""
        if not isinstance(value, list):
            return False
        return any(item in value for item in items)

    def _array_contains_all(self, value: Any, items: List) -> bool:
        """Check if array contains all items."""
        if not isinstance(value, list):
            return False
        return all(item in value for item in items)

    # Date operation helpers
    def _date_before(self, value: Any, target_date: datetime) -> bool:
        """Check if date is before target."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return False
        if isinstance(value, datetime):
            return value < target_date
        return False

    def _date_after(self, value: Any, target_date: datetime) -> bool:
        """Check if date is after target."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return False
        if isinstance(value, datetime):
            return value > target_date
        return False

    def _date_between(
        self, value: Any, date_range: List[datetime]
    ) -> bool:
        """Check if date is between range."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return False
        if isinstance(value, datetime) and len(date_range) == 2:
            return date_range[0] <= value <= date_range[1]
        return False