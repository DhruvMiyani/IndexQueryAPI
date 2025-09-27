"""
Repository layer for Vector Database API.

This module implements the repository pattern for data access,
following Domain-Driven Design principles.
"""

from .base import BaseRepository, RepositoryException, NotFoundError, AlreadyExistsError
from .chunk_repository import ChunkRepository, InMemoryChunkRepository
from .document_repository import DocumentRepository, InMemoryDocumentRepository
from .library_repository import LibraryRepository, InMemoryLibraryRepository

__all__ = [
    "BaseRepository",
    "RepositoryException",
    "NotFoundError",
    "AlreadyExistsError",
    "ChunkRepository",
    "InMemoryChunkRepository",
    "DocumentRepository",
    "InMemoryDocumentRepository",
    "LibraryRepository",
    "InMemoryLibraryRepository",
]