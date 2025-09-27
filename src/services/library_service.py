"""
Library service for business logic.

Handles library lifecycle, statistics, and cascade operations.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from models.library import Library, LibraryCreate, LibraryUpdate
from repository import LibraryRepository, DocumentRepository, ChunkRepository


class LibraryService:
    """
    Service for library operations.

    Orchestrates library management and related operations.
    """

    def __init__(
        self,
        library_repo: LibraryRepository,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
    ):
        """
        Initialize library service.

        Args:
            library_repo: Library repository instance
            document_repo: Document repository instance
            chunk_repo: Chunk repository instance
        """
        self.library_repo = library_repo
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo

    async def create_library(self, library_data: LibraryCreate) -> Library:
        """
        Create a new library.

        Args:
            library_data: Library creation data

        Returns:
            Created library
        """
        # Generate ID and timestamps
        library = Library(
            id=uuid4(),
            name=library_data.name,
            metadata=library_data.metadata,
            document_count=0,
            chunk_count=0,
            index_status="none"
        )

        # Save to repository
        return await self.library_repo.create(library)

    async def get_library(self, library_id: UUID) -> Library:
        """
        Get library by ID.

        Args:
            library_id: Library ID

        Returns:
            Library instance
        """
        return await self.library_repo.get(library_id)

    async def list_libraries(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Library]:
        """
        List all libraries with pagination.

        Args:
            limit: Maximum number to return
            offset: Number to skip

        Returns:
            List of libraries
        """
        return await self.library_repo.list(limit=limit, offset=offset)

    async def update_library(
        self, library_id: UUID, update_data: LibraryUpdate
    ) -> Library:
        """
        Update library metadata.

        Args:
            library_id: Library ID
            update_data: Update data

        Returns:
            Updated library
        """
        library = await self.library_repo.get(library_id)

        # Update fields
        if update_data.name is not None:
            library.name = update_data.name

        if update_data.metadata is not None:
            library.metadata = update_data.metadata
            library.metadata.updated_at = datetime.utcnow()

        return await self.library_repo.update(library_id, library)

    async def delete_library(self, library_id: UUID) -> bool:
        """
        Delete library and all its contents (cascade).

        Args:
            library_id: Library ID

        Returns:
            True if deleted
        """
        # Delete all chunks in library
        documents = await self.document_repo.get_by_library(library_id)
        for doc in documents:
            await self.chunk_repo.delete_by_document(doc.id)

        # Delete all documents in library
        await self.document_repo.delete_by_library(library_id)

        # Delete library itself
        return await self.library_repo.delete(library_id)

    async def get_library_statistics(self, library_id: UUID) -> dict:
        """
        Get detailed statistics for a library.

        Args:
            library_id: Library ID

        Returns:
            Statistics dictionary
        """
        library = await self.library_repo.get(library_id)
        documents = await self.document_repo.get_by_library(library_id)

        # Calculate statistics
        total_chunks = sum(doc.chunk_count for doc in documents)
        avg_chunks_per_doc = (
            total_chunks / len(documents) if documents else 0
        )

        return {
            "library_id": library_id,
            "name": library.name,
            "document_count": library.document_count,
            "chunk_count": library.chunk_count,
            "avg_chunks_per_document": avg_chunks_per_doc,
            "index_status": library.index_status,
            "index_type": library.index_type,
            "created_at": library.metadata.created_at,
            "updated_at": library.metadata.updated_at,
            "vector_dimension": library.metadata.vector_dimension,
        }

    async def update_library_counts(
        self, library_id: UUID, doc_delta: int = 0, chunk_delta: int = 0
    ) -> None:
        """
        Update library document and chunk counts.

        Args:
            library_id: Library ID
            doc_delta: Change in document count
            chunk_delta: Change in chunk count
        """
        library = await self.library_repo.get(library_id)
        library.document_count += doc_delta
        library.chunk_count += chunk_delta
        library.metadata.updated_at = datetime.utcnow()
        await self.library_repo.update(library_id, library)