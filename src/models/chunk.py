"""
Chunk-related Pydantic models for the Vector Database API.

A Chunk represents a piece of text with an associated vector embedding and metadata.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for a chunk with fixed schema."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    position: int = Field(
        description="Position of chunk within the document", ge=0
    )
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    author: Optional[str] = Field(None, description="Author of the content")
    language: Optional[str] = Field(None, description="Content language code")
    source: Optional[str] = Field(None, description="Source URL or file path")


class ChunkBase(BaseModel):
    """Base chunk model with common fields."""

    text: str = Field(description="The text content of the chunk", min_length=1)
    document_id: UUID = Field(description="ID of the parent document")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class ChunkCreate(ChunkBase):
    """Model for creating a new chunk."""

    embedding: Optional[List[float]] = Field(
        None,
        description="Pre-computed embedding vector. If not provided, will be generated from text",
    )


class ChunkUpdate(BaseModel):
    """Model for updating an existing chunk."""

    text: Optional[str] = Field(None, min_length=1)
    metadata: Optional[ChunkMetadata] = None
    embedding: Optional[List[float]] = Field(
        None,
        description="Updated embedding vector. If text is changed, this will be regenerated",
    )


class Chunk(ChunkBase):
    """Complete chunk model with all fields."""

    id: UUID = Field(description="Unique identifier for the chunk")
    embedding: List[float] = Field(description="Vector embedding of the text")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "text": "This is a sample chunk of text content.",
                "document_id": "660e8400-e29b-41d4-a716-446655440001",
                "embedding": [0.1, 0.2, -0.3, 0.4, -0.5],
                "metadata": {
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": "2023-01-01T00:00:00",
                    "position": 0,
                    "tags": ["introduction", "overview"],
                    "author": "John Doe",
                    "language": "en",
                    "source": "document.pdf"
                }
            }
        }