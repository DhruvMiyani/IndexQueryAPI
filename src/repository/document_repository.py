"""
Repository for Document entities.

Handles data access for documents following the repository pattern.
"""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID

from models.document import Document
from .base import BaseRepository, NotFoundError, AlreadyExistsError


class DocumentRepository(BaseRepository[Document]):
    """Abstract interface for Document repository."""

    async def get_by_library(
        self, library_id: UUID, offset: int = 0, limit: int = 100
    ) -> List[Document]:
        """
        Get documents in a library with pagination.

        Args:
            library_id: The library ID
            offset: Number of documents to skip
            limit: Maximum documents to return

        Returns:
            List of documents in the library
        """
        raise NotImplementedError

    async def delete_by_library(self, library_id: UUID) -> int:
        """
        Delete all documents in a library.

        Args:
            library_id: The library ID

        Returns:
            Number of documents deleted
        """
        raise NotImplementedError


class InMemoryDocumentRepository(DocumentRepository):
    """
    In-memory implementation of Document repository.

    Thread-safe implementation using asyncio.Lock.
    """

    def __init__(self):
        """Initialize empty repository with lock for thread safety."""
        self._documents: Dict[UUID, Document] = {}
        self._library_index: Dict[UUID, List[UUID]] = {}
        self._lock = asyncio.Lock()

    async def get(self, entity_id: UUID) -> Document:
        """Get document by ID."""
        async with self._lock:
            if entity_id not in self._documents:
                raise NotFoundError("Document", entity_id)
            return self._documents[entity_id]

    async def list(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        """List documents with pagination."""
        async with self._lock:
            documents = list(self._documents.values())

            # Sort by creation time for consistent ordering
            documents.sort(key=lambda x: x.metadata.created_at)

            # Apply pagination
            if limit is not None:
                documents = documents[offset : offset + limit]
            else:
                documents = documents[offset:]

            return documents

    async def create(self, entity: Document) -> Document:
        """Create a new document."""
        async with self._lock:
            # Check if ID already exists
            if entity.id in self._documents:
                raise AlreadyExistsError("Document", entity.id)

            # Store document
            self._documents[entity.id] = entity

            # Update library index
            if entity.library_id not in self._library_index:
                self._library_index[entity.library_id] = []
            self._library_index[entity.library_id].append(entity.id)

            return entity

    async def update(self, entity_id: UUID, entity: Document) -> Document:
        """Update an existing document."""
        async with self._lock:
            if entity_id not in self._documents:
                raise NotFoundError("Document", entity_id)

            old_document = self._documents[entity_id]

            # If library changed, update indexes
            if old_document.library_id != entity.library_id:
                # Remove from old library index
                if old_document.library_id in self._library_index:
                    self._library_index[old_document.library_id].remove(entity_id)
                    if not self._library_index[old_document.library_id]:
                        del self._library_index[old_document.library_id]

                # Add to new library index
                if entity.library_id not in self._library_index:
                    self._library_index[entity.library_id] = []
                self._library_index[entity.library_id].append(entity_id)

            # Update document
            self._documents[entity_id] = entity

            return entity

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a document."""
        async with self._lock:
            if entity_id not in self._documents:
                raise NotFoundError("Document", entity_id)

            document = self._documents[entity_id]

            # Remove from library index
            if document.library_id in self._library_index:
                self._library_index[document.library_id].remove(entity_id)
                if not self._library_index[document.library_id]:
                    del self._library_index[document.library_id]

            # Remove document
            del self._documents[entity_id]

            return True

    async def exists(self, entity_id: UUID) -> bool:
        """Check if document exists."""
        async with self._lock:
            return entity_id in self._documents

    async def count(self) -> int:
        """Get total count of documents."""
        async with self._lock:
            return len(self._documents)

    async def get_by_library(
        self, library_id: UUID, offset: int = 0, limit: int = 100
    ) -> List[Document]:
        """Get documents in a library with pagination."""
        async with self._lock:
            document_ids = self._library_index.get(library_id, [])
            documents = [self._documents[doc_id] for doc_id in document_ids]
            # Sort by creation time
            documents.sort(key=lambda x: x.metadata.created_at)
            return documents[offset : offset + limit]

    async def delete_by_library(self, library_id: UUID) -> int:
        """Delete all documents in a library."""
        async with self._lock:
            document_ids = self._library_index.get(library_id, [])
            count = len(document_ids)

            # Delete each document
            for doc_id in document_ids:
                del self._documents[doc_id]

            # Clear library index
            if library_id in self._library_index:
                del self._library_index[library_id]

            return count