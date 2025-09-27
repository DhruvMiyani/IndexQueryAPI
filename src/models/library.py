"""
Library-related Pydantic models for the Vector Database API.

A Library is a collection of documents and can contain its own metadata.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LibraryMetadata(BaseModel):
    """Metadata for a library with fixed schema."""

    description: Optional[str] = Field(None, description="Library description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    owner: Optional[str] = Field(None, description="Library owner")
    tags: List[str] = Field(default_factory=list, description="Library tags")
    category: Optional[str] = Field(None, description="Library category")
    is_public: bool = Field(default=False, description="Whether library is public")
    embedding_model: Optional[str] = Field(
        None, description="Model used for generating embeddings"
    )
    vector_dimension: Optional[int] = Field(
        None, description="Dimension of vectors in this library", ge=1
    )


class LibraryBase(BaseModel):
    """Base library model with common fields."""

    name: str = Field(description="Library name", min_length=1, max_length=255)
    metadata: LibraryMetadata = Field(default_factory=LibraryMetadata)


class LibraryCreate(LibraryBase):
    """Model for creating a new library."""

    pass


class LibraryUpdate(BaseModel):
    """Model for updating an existing library."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[LibraryMetadata] = None


class Library(LibraryBase):
    """Complete library model with all fields."""

    id: UUID = Field(description="Unique identifier for the library")
    document_ids: List[UUID] = Field(
        default_factory=list,
        description="List of document IDs in this library"
    )
    document_count: int = Field(
        default=0,
        description="Total number of documents in this library",
        ge=0
    )
    chunk_count: int = Field(
        default=0,
        description="Total number of chunks across all documents",
        ge=0
    )
    index_type: Optional[str] = Field(
        None,
        description="Type of index built for this library (linear, kd_tree, lsh)"
    )
    index_status: str = Field(
        default="none",
        description="Status of index (none, building, ready, error)"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "770e8400-e29b-41d4-a716-446655440002",
                "name": "Research Papers",
                "document_ids": [
                    "660e8400-e29b-41d4-a716-446655440001",
                    "660e8400-e29b-41d4-a716-446655440004"
                ],
                "document_count": 2,
                "chunk_count": 15,
                "index_type": "kd_tree",
                "index_status": "ready",
                "metadata": {
                    "description": "Collection of AI research papers",
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": "2023-01-01T00:00:00",
                    "owner": "research_team",
                    "tags": ["ai", "research", "papers"],
                    "category": "academic",
                    "is_public": False,
                    "embedding_model": "cohere-embed-english-v2.0",
                    "vector_dimension": 1024
                }
            }
        }