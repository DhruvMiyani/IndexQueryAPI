"""
Document-related Pydantic models for the Vector Database API.

A Document is made up of multiple chunks and contains its own metadata.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document with fixed schema."""

    title: str = Field(description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = Field(None, description="Original source file or URL")
    file_type: Optional[str] = Field(None, description="File type (pdf, txt, etc.)")
    file_size: Optional[int] = Field(None, description="File size in bytes", ge=0)
    tags: List[str] = Field(default_factory=list, description="Document tags")
    category: Optional[str] = Field(None, description="Document category")
    language: Optional[str] = Field(None, description="Primary language code")


class DocumentBase(BaseModel):
    """Base document model with common fields."""

    library_id: UUID = Field(description="ID of the parent library")
    metadata: DocumentMetadata


class DocumentCreate(DocumentBase):
    """Model for creating a new document."""

    pass


class DocumentUpdate(BaseModel):
    """Model for updating an existing document."""

    metadata: Optional[DocumentMetadata] = None


class Document(DocumentBase):
    """Complete document model with all fields."""

    id: UUID = Field(description="Unique identifier for the document")
    chunk_ids: List[UUID] = Field(
        default_factory=list,
        description="List of chunk IDs that belong to this document"
    )
    chunk_count: int = Field(
        default=0,
        description="Total number of chunks in this document",
        ge=0
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "660e8400-e29b-41d4-a716-446655440001",
                "library_id": "770e8400-e29b-41d4-a716-446655440002",
                "chunk_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400-e29b-41d4-a716-446655440003"
                ],
                "chunk_count": 2,
                "metadata": {
                    "title": "Sample Document",
                    "author": "John Doe",
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": "2023-01-01T00:00:00",
                    "source": "sample.pdf",
                    "file_type": "pdf",
                    "file_size": 1024000,
                    "tags": ["research", "ai"],
                    "category": "technical",
                    "language": "en"
                }
            }
        }