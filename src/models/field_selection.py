"""
Field selection models for reducing API payload sizes.

Allows clients to specify which fields they want in responses.
"""

from typing import Set, Optional, List, Any, Dict
from pydantic import BaseModel, Field, validator


class FieldSelector:
    """Utility class for field selection on Pydantic models."""

    @staticmethod
    def select_fields(model_instance: BaseModel, fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Select specific fields from a Pydantic model instance.

        Args:
            model_instance: The Pydantic model instance
            fields: Set of field names to include. If None, includes all fields.

        Returns:
            Dictionary with only the selected fields
        """
        if fields is None:
            return model_instance.model_dump()

        model_dict = model_instance.model_dump()
        return {key: value for key, value in model_dict.items() if key in fields}

    @staticmethod
    def parse_fields_query(fields_str: Optional[str]) -> Optional[Set[str]]:
        """
        Parse comma-separated fields query parameter.

        Args:
            fields_str: Comma-separated field names

        Returns:
            Set of field names or None if empty
        """
        if not fields_str:
            return None

        fields = {field.strip() for field in fields_str.split(",") if field.strip()}
        return fields if fields else None

    @staticmethod
    def validate_fields(fields: Set[str], model_class: type) -> Set[str]:
        """
        Validate that requested fields exist on the model.

        Args:
            fields: Set of field names to validate
            model_class: The Pydantic model class

        Returns:
            Set of valid field names

        Raises:
            ValueError: If any field is invalid
        """
        if not hasattr(model_class, 'model_fields'):
            return fields

        model_fields = set(model_class.model_fields.keys())
        invalid_fields = fields - model_fields

        if invalid_fields:
            raise ValueError(f"Invalid fields: {', '.join(invalid_fields)}. "
                           f"Available fields: {', '.join(sorted(model_fields))}")

        return fields


class FieldSelectionMixin(BaseModel):
    """Mixin to add field selection capability to response models."""

    def select(self, fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Select specific fields from this model instance."""
        return FieldSelector.select_fields(self, fields)