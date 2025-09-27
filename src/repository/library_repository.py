"""
Repository for Library entities.

Handles data access for libraries following the repository pattern.
"""

import asyncio
from typing import Dict, List, Optional
from uuid import UUID

from models.library import Library, LibraryCreate, LibraryUpdate
from .base import BaseRepository, NotFoundError, AlreadyExistsError


class LibraryRepository(BaseRepository[Library]):
    """Abstract interface for Library repository."""

    async def get_by_name(self, name: str) -> Optional[Library]:
        """
        Get a library by name.

        Args:
            name: The library name

        Returns:
            The library if found, None otherwise
        """
        raise NotImplementedError


class InMemoryLibraryRepository(LibraryRepository):
    """
    In-memory implementation of Library repository.

    Thread-safe implementation using asyncio.Lock.
    """

    def __init__(self):
        """Initialize empty repository with lock for thread safety."""
        self._libraries: Dict[UUID, Library] = {}
        self._name_index: Dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def get(self, entity_id: UUID) -> Library:
        """Get library by ID."""
        async with self._lock:
            if entity_id not in self._libraries:
                raise NotFoundError("Library", entity_id)
            return self._libraries[entity_id]

    async def list(self, limit: Optional[int] = None, offset: int = 0) -> List[Library]:
        """List libraries with pagination."""
        async with self._lock:
            libraries = list(self._libraries.values())

            # Sort by creation time for consistent ordering
            libraries.sort(key=lambda x: x.metadata.created_at)

            # Apply pagination
            if limit is not None:
                libraries = libraries[offset : offset + limit]
            else:
                libraries = libraries[offset:]

            return libraries

    async def create(self, entity: Library) -> Library:
        """Create a new library."""
        async with self._lock:
            # Check if ID already exists
            if entity.id in self._libraries:
                raise AlreadyExistsError("Library", entity.id)

            # Check if name already exists (enforce unique names)
            if entity.name in self._name_index:
                raise RepositoryException(
                    f"Library with name '{entity.name}' already exists"
                )

            # Store library
            self._libraries[entity.id] = entity
            self._name_index[entity.name] = entity.id

            return entity

    async def update(self, entity_id: UUID, entity: Library) -> Library:
        """Update an existing library."""
        async with self._lock:
            if entity_id not in self._libraries:
                raise NotFoundError("Library", entity_id)

            old_library = self._libraries[entity_id]

            # If name changed, update name index
            if old_library.name != entity.name:
                # Check new name isn't taken
                if entity.name in self._name_index:
                    raise RepositoryException(
                        f"Library with name '{entity.name}' already exists"
                    )
                # Update name index
                del self._name_index[old_library.name]
                self._name_index[entity.name] = entity_id

            # Update library
            self._libraries[entity_id] = entity

            return entity

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a library."""
        async with self._lock:
            if entity_id not in self._libraries:
                raise NotFoundError("Library", entity_id)

            library = self._libraries[entity_id]

            # Remove from indexes
            del self._libraries[entity_id]
            del self._name_index[library.name]

            return True

    async def exists(self, entity_id: UUID) -> bool:
        """Check if library exists."""
        async with self._lock:
            return entity_id in self._libraries

    async def count(self) -> int:
        """Get total count of libraries."""
        async with self._lock:
            return len(self._libraries)

    async def get_by_name(self, name: str) -> Optional[Library]:
        """Get library by name."""
        async with self._lock:
            library_id = self._name_index.get(name)
            if library_id:
                return self._libraries[library_id]
            return None


from .base import RepositoryException  # Import after class definition to avoid circular import