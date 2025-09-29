"""
Index service for managing vector indexes.

Handles index lifecycle and algorithm selection.
"""

from typing import Dict, List, Optional
from uuid import UUID

from indexes import BaseIndex, IndexType
from indexes.advanced_index_factory import AdvancedIndexFactory
from repository import ChunkRepository, LibraryRepository


class IndexService:
    """
    Service for index operations.

    Manages vector indexes for libraries.
    """

    def __init__(
        self,
        library_repo: LibraryRepository,
        chunk_repo: ChunkRepository,
    ):
        """
        Initialize index service.

        Args:
            library_repo: Library repository instance
            chunk_repo: Chunk repository instance
        """
        self.library_repo = library_repo
        self.chunk_repo = chunk_repo
        self._indexes: Dict[UUID, BaseIndex] = {}

    async def build_index(
        self,
        library_id: UUID,
        index_type: Optional[IndexType] = None,
        force_rebuild: bool = False,
    ) -> BaseIndex:
        """
        Build or rebuild index for a library.

        Args:
            library_id: Library ID
            index_type: Type of index to build (auto-select if None)
            force_rebuild: Force rebuild even if index exists

        Returns:
            Built index instance
        """
        library = await self.library_repo.get(library_id)

        # Check if index exists and rebuild not forced
        if library_id in self._indexes and not force_rebuild:
            return self._indexes[library_id]

        # Get all vectors in library
        vectors = await self.chunk_repo.get_vectors_by_library(library_id)

        if not vectors:
            raise ValueError(f"Library {library_id} has no chunks to index")

        # Determine vector dimension
        dimension = len(vectors[0][1])

        # Auto-select index type if not provided
        if index_type is None:
            index_type = AdvancedIndexFactory.recommend_index_type(
                dimension=dimension,
                dataset_size=len(vectors),
                accuracy_required=True,
            )

        # Update library status
        library.index_status = "building"
        await self.library_repo.update(library_id, library)

        try:
            # Create index
            index = AdvancedIndexFactory.create(index_type, dimension)

            # Build index with vectors
            index.build(vectors)

            # Store index
            self._indexes[library_id] = index

            # Update library
            library.index_type = index_type.value
            library.index_status = "ready"
            library.metadata.vector_dimension = dimension
            await self.library_repo.update(library_id, library)

            return index

        except Exception as e:
            # Update library status on error
            library.index_status = "error"
            await self.library_repo.update(library_id, library)
            raise ValueError(f"Failed to build index: {e}")

    async def get_index(self, library_id: UUID) -> Optional[BaseIndex]:
        """
        Get index for a library.

        Args:
            library_id: Library ID

        Returns:
            Index instance or None if not built
        """
        if library_id not in self._indexes:
            # Try to build index if not exists
            library = await self.library_repo.get(library_id)
            if library.index_status == "ready" and library.chunk_count > 0:
                # Rebuild from stored vectors
                await self.build_index(library_id)

        return self._indexes.get(library_id)

    async def delete_index(self, library_id: UUID) -> bool:
        """
        Delete index for a library.

        Args:
            library_id: Library ID

        Returns:
            True if deleted
        """
        if library_id in self._indexes:
            del self._indexes[library_id]

            # Update library
            library = await self.library_repo.get(library_id)
            library.index_type = None
            library.index_status = "none"
            await self.library_repo.update(library_id, library)

            return True
        return False

    async def update_index(
        self, library_id: UUID, chunk_id: UUID, vector: List[float]
    ) -> None:
        """
        Add or update a vector in the index.

        Args:
            library_id: Library ID
            chunk_id: Chunk ID
            vector: Embedding vector
        """
        index = await self.get_index(library_id)
        if index:
            index.add(chunk_id, vector)

    async def remove_from_index(
        self, library_id: UUID, chunk_id: UUID
    ) -> bool:
        """
        Remove a vector from the index.

        Args:
            library_id: Library ID
            chunk_id: Chunk ID

        Returns:
            True if removed
        """
        index = await self.get_index(library_id)
        if index:
            return index.remove(chunk_id)
        return False

    async def get_index_stats(self, library_id: UUID) -> dict:
        """
        Get statistics for a library's index.

        Args:
            library_id: Library ID

        Returns:
            Index statistics
        """
        library = await self.library_repo.get(library_id)
        index = await self.get_index(library_id)

        stats = {
            "library_id": library_id,
            "index_status": library.index_status,
            "index_type": library.index_type,
        }

        if index:
            stats.update(index.get_stats())

        return stats

    def clear_all_indexes(self) -> None:
        """Clear all indexes from memory."""
        self._indexes.clear()


