"""
FastAPI dependencies for dependency injection.

Provides service instances and common utilities.
"""

from repository import (
    InMemoryLibraryRepository,
    InMemoryDocumentRepository,
    InMemoryChunkRepository,
)
from services import (
    LibraryService,
    DocumentService,
    ChunkService,
    IndexService,
    SearchService,
    EmbeddingService,
)

# Create repository instances (singletons for in-memory storage)
_library_repo = InMemoryLibraryRepository()
_document_repo = InMemoryDocumentRepository()
_chunk_repo = InMemoryChunkRepository()

# Create service instances
_embedding_service = EmbeddingService(provider="cohere", dimension=1024)
_library_service = LibraryService(_library_repo, _document_repo, _chunk_repo)
_document_service = DocumentService(
    _document_repo, _chunk_repo, _library_service, _embedding_service
)
_chunk_service = ChunkService(_chunk_repo, _embedding_service, _library_service)
_index_service = IndexService(_library_repo, _chunk_repo)
_search_service = SearchService(
    _chunk_repo, _document_repo, _library_repo, _index_service, _embedding_service
)


def get_library_service() -> LibraryService:
    """Get library service instance."""
    return _library_service


def get_document_service() -> DocumentService:
    """Get document service instance."""
    return _document_service


def get_chunk_service() -> ChunkService:
    """Get chunk service instance."""
    return _chunk_service


def get_index_service() -> IndexService:
    """Get index service instance."""
    return _index_service


def get_search_service() -> SearchService:
    """Get search service instance."""
    return _search_service


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return _embedding_service