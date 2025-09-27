"""
Repository for Chunk entities.

Handles data access for chunks following the repository pattern.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from models.chunk import Chunk
from .base import BaseRepository, NotFoundError, AlreadyExistsError


class ChunkRepository(BaseRepository[Chunk]):
    """Abstract interface for Chunk repository."""

    async def get_by_document(self, document_id: UUID) -> List[Chunk]:
        """
        Get all chunks in a document.

        Args:
            document_id: The document ID

        Returns:
            List of chunks in the document
        """
        raise NotImplementedError

    async def get_by_library(
        self, library_id: UUID, offset: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """
        Get chunks in a library with pagination.

        Args:
            library_id: The library ID
            offset: Number of chunks to skip
            limit: Maximum chunks to return

        Returns:
            List of chunks in the library
        """
        raise NotImplementedError

    async def delete_by_document(self, document_id: UUID) -> int:
        """
        Delete all chunks in a document.

        Args:
            document_id: The document ID

        Returns:
            Number of chunks deleted
        """
        raise NotImplementedError

    async def get_vectors_by_library(self, library_id: UUID) -> List[Tuple[UUID, List[float]]]:
        """
        Get all chunk vectors in a library.

        Args:
            library_id: The library ID

        Returns:
            List of tuples (chunk_id, embedding_vector)
        """
        raise NotImplementedError


class InMemoryChunkRepository(ChunkRepository):
    """
    In-memory implementation of Chunk repository.

    Thread-safe implementation using asyncio.Lock.
    """

    def __init__(self):
        """Initialize empty repository with lock for thread safety."""
        self._chunks: Dict[UUID, Chunk] = {}
        self._document_index: Dict[UUID, List[UUID]] = {}
        self._library_chunks: Dict[UUID, List[UUID]] = {}
        self._lock = asyncio.Lock()

    async def get(self, entity_id: UUID) -> Chunk:
        """Get chunk by ID."""
        async with self._lock:
            if entity_id not in self._chunks:
                raise NotFoundError("Chunk", entity_id)
            return self._chunks[entity_id]

    async def list(self, limit: Optional[int] = None, offset: int = 0) -> List[Chunk]:
        """List chunks with pagination."""
        async with self._lock:
            chunks = list(self._chunks.values())

            # Sort by creation time for consistent ordering
            chunks.sort(key=lambda x: x.metadata.created_at)

            # Apply pagination
            if limit is not None:
                chunks = chunks[offset : offset + limit]
            else:
                chunks = chunks[offset:]

            return chunks

    async def create(self, entity: Chunk) -> Chunk:
        """Create a new chunk."""
        async with self._lock:
            # Check if ID already exists
            if entity.id in self._chunks:
                raise AlreadyExistsError("Chunk", entity.id)

            # Store chunk
            self._chunks[entity.id] = entity

            # Update document index
            if entity.document_id not in self._document_index:
                self._document_index[entity.document_id] = []
            self._document_index[entity.document_id].append(entity.id)

            return entity

    async def update(self, entity_id: UUID, entity: Chunk) -> Chunk:
        """Update an existing chunk."""
        async with self._lock:
            if entity_id not in self._chunks:
                raise NotFoundError("Chunk", entity_id)

            old_chunk = self._chunks[entity_id]

            # If document changed, update indexes
            if old_chunk.document_id != entity.document_id:
                # Remove from old document index
                if old_chunk.document_id in self._document_index:
                    self._document_index[old_chunk.document_id].remove(entity_id)
                    if not self._document_index[old_chunk.document_id]:
                        del self._document_index[old_chunk.document_id]

                # Add to new document index
                if entity.document_id not in self._document_index:
                    self._document_index[entity.document_id] = []
                self._document_index[entity.document_id].append(entity_id)

            # Update chunk
            self._chunks[entity_id] = entity

            return entity

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a chunk."""
        async with self._lock:
            if entity_id not in self._chunks:
                raise NotFoundError("Chunk", entity_id)

            chunk = self._chunks[entity_id]

            # Remove from document index
            if chunk.document_id in self._document_index:
                self._document_index[chunk.document_id].remove(entity_id)
                if not self._document_index[chunk.document_id]:
                    del self._document_index[chunk.document_id]

            # Remove from library chunks if tracked
            for lib_id in list(self._library_chunks.keys()):
                if entity_id in self._library_chunks[lib_id]:
                    self._library_chunks[lib_id].remove(entity_id)
                    if not self._library_chunks[lib_id]:
                        del self._library_chunks[lib_id]

            # Remove chunk
            del self._chunks[entity_id]

            return True

    async def exists(self, entity_id: UUID) -> bool:
        """Check if chunk exists."""
        async with self._lock:
            return entity_id in self._chunks

    async def count(self) -> int:
        """Get total count of chunks."""
        async with self._lock:
            return len(self._chunks)

    async def get_by_document(self, document_id: UUID) -> List[Chunk]:
        """Get all chunks in a document."""
        async with self._lock:
            chunk_ids = self._document_index.get(document_id, [])
            chunks = [self._chunks[chunk_id] for chunk_id in chunk_ids]
            # Sort by position in document
            chunks.sort(key=lambda x: x.metadata.position)
            return chunks

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks in a document."""
        async with self._lock:
            chunk_ids = self._document_index.get(document_id, [])
            count = len(chunk_ids)

            # Delete each chunk
            for chunk_id in chunk_ids:
                # Remove from library chunks if tracked
                for lib_id in list(self._library_chunks.keys()):
                    if chunk_id in self._library_chunks[lib_id]:
                        self._library_chunks[lib_id].remove(chunk_id)

                del self._chunks[chunk_id]

            # Clear document index
            if document_id in self._document_index:
                del self._document_index[document_id]

            return count

    async def get_by_library(
        self, library_id: UUID, offset: int = 0, limit: int = 100
    ) -> List[Chunk]:
        """
        Get chunks in a library with pagination.

        Note: This requires tracking library associations separately
        or joining through documents. For now, we'll track it.
        """
        async with self._lock:
            chunk_ids = self._library_chunks.get(library_id, [])
            chunks = [self._chunks[chunk_id] for chunk_id in chunk_ids]
            chunks.sort(key=lambda x: x.metadata.created_at)
            return chunks[offset : offset + limit]

    async def add_to_library_index(self, library_id: UUID, chunk_id: UUID) -> None:
        """
        Add a chunk to library index.

        This is called when a chunk is added to a document in a library.
        """
        async with self._lock:
            if library_id not in self._library_chunks:
                self._library_chunks[library_id] = []
            if chunk_id not in self._library_chunks[library_id]:
                self._library_chunks[library_id].append(chunk_id)

    async def get_vectors_by_library(self, library_id: UUID) -> List[Tuple[UUID, List[float]]]:
        """Get all chunk vectors in a library for indexing."""
        async with self._lock:
            chunk_ids = self._library_chunks.get(library_id, [])
            vectors = []
            for chunk_id in chunk_ids:
                if chunk_id in self._chunks:
                    chunk = self._chunks[chunk_id]
                    vectors.append((chunk_id, chunk.embedding))
            return vectors