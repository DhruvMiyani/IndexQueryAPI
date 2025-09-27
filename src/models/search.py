"""
Search-related Pydantic models for the Vector Database API.

Models for handling vector search requests and responses.
"""

from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Model for vector search requests."""

    query_text: Optional[str] = Field(
        None,
        description="Text query to be converted to embedding",
        min_length=1
    )
    query_vector: Optional[List[float]] = Field(
        None,
        description="Pre-computed query vector"
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
        ge=1,
        le=1000
    )
    filters: Optional[Dict[str, Union[str, int, float, bool, List[str]]]] = Field(
        None,
        description="Metadata filters to apply to search"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in results"
    )
    include_text: bool = Field(
        default=True,
        description="Whether to include text content in results"
    )
    min_score: Optional[float] = Field(
        None,
        description="Minimum similarity score threshold",
        ge=0.0,
        le=1.0
    )

    def model_post_init(self, __context) -> None:
        """Validate that either query_text or query_vector is provided."""
        if not self.query_text and not self.query_vector:
            raise ValueError("Either query_text or query_vector must be provided")
        if self.query_text and self.query_vector:
            raise ValueError("Only one of query_text or query_vector should be provided")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "query_text": "machine learning algorithms",
                "top_k": 5,
                "filters": {
                    "author": "John Doe",
                    "tags": ["ai", "research"],
                    "language": "en"
                },
                "include_metadata": True,
                "include_text": True,
                "min_score": 0.7
            }
        }


class SearchResult(BaseModel):
    """Model for individual search results."""

    chunk_id: UUID = Field(description="ID of the matching chunk")
    document_id: UUID = Field(description="ID of the parent document")
    score: float = Field(
        description="Similarity score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    text: Optional[str] = Field(None, description="Text content of the chunk")
    metadata: Optional[Dict] = Field(None, description="Chunk metadata")
    document_metadata: Optional[Dict] = Field(
        None,
        description="Parent document metadata"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_id": "660e8400-e29b-41d4-a716-446655440001",
                "score": 0.8532,
                "text": "Machine learning is a subset of artificial intelligence...",
                "metadata": {
                    "position": 0,
                    "tags": ["introduction"],
                    "author": "John Doe"
                },
                "document_metadata": {
                    "title": "Introduction to ML",
                    "category": "technical"
                }
            }
        }


class SearchResponse(BaseModel):
    """Model for search response containing results and metadata."""

    query: str = Field(description="The original query")
    results: List[SearchResult] = Field(description="List of search results")
    total_results: int = Field(
        description="Total number of results found",
        ge=0
    )
    search_time_ms: float = Field(
        description="Time taken to execute search in milliseconds",
        ge=0
    )
    index_type: Optional[str] = Field(
        None,
        description="Type of index used for search"
    )
    filters_applied: Optional[Dict] = Field(
        None,
        description="Filters that were applied to the search"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "results": [
                    {
                        "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_id": "660e8400-e29b-41d4-a716-446655440001",
                        "score": 0.8532,
                        "text": "Machine learning is a subset of artificial intelligence...",
                        "metadata": {"position": 0, "tags": ["introduction"]},
                        "document_metadata": {"title": "Introduction to ML"}
                    }
                ],
                "total_results": 1,
                "search_time_ms": 15.2,
                "index_type": "kd_tree",
                "filters_applied": {"language": "en"}
            }
        }