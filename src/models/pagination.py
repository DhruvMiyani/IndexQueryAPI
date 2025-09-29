"""
Pagination models for standardized API responses.

Following Azure best practices for RESTful APIs.
"""

from typing import Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Standard pagination parameters for list endpoints."""

    limit: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of items to return (1-100)"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip"
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response wrapper."""

    items: List[T] = Field(description="List of items in current page")
    total: int = Field(description="Total number of items available")
    limit: int = Field(description="Maximum items per page")
    offset: int = Field(description="Number of items skipped")
    has_next: bool = Field(description="Whether more items are available")
    has_previous: bool = Field(description="Whether previous items are available")

    @property
    def next_offset(self) -> Optional[int]:
        """Calculate next offset for pagination."""
        if self.has_next:
            return self.offset + self.limit
        return None

    @property
    def previous_offset(self) -> Optional[int]:
        """Calculate previous offset for pagination."""
        if self.has_previous:
            return max(0, self.offset - self.limit)
        return None

    class Config:
        """Pydantic configuration."""
        from_attributes = True