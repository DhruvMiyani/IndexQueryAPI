"""
Base repository interface and exceptions.

Defines the abstract interface for all repositories following DDD patterns.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from uuid import UUID


# Generic type for domain models
T = TypeVar("T")


class RepositoryException(Exception):
    """Base exception for repository operations."""
    pass


class NotFoundError(RepositoryException):
    """Raised when an entity is not found."""

    def __init__(self, entity_type: str, entity_id: UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID {entity_id} not found")


class AlreadyExistsError(RepositoryException):
    """Raised when attempting to create an entity that already exists."""

    def __init__(self, entity_type: str, entity_id: UUID):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID {entity_id} already exists")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository interface.

    Defines standard CRUD operations for all repositories.
    """

    @abstractmethod
    async def get(self, entity_id: UUID) -> T:
        """
        Get an entity by ID.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            The entity if found

        Raises:
            NotFoundError: If entity does not exist
        """
        pass

    @abstractmethod
    async def list(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        List entities with pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip

        Returns:
            List of entities
        """
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.

        Args:
            entity: The entity to create

        Returns:
            The created entity

        Raises:
            AlreadyExistsError: If entity ID already exists
        """
        pass

    @abstractmethod
    async def update(self, entity_id: UUID, entity: T) -> T:
        """
        Update an existing entity.

        Args:
            entity_id: The ID of the entity to update
            entity: The updated entity data

        Returns:
            The updated entity

        Raises:
            NotFoundError: If entity does not exist
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: The ID of the entity to delete

        Returns:
            True if deleted successfully

        Raises:
            NotFoundError: If entity does not exist
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """
        Check if an entity exists.

        Args:
            entity_id: The ID to check

        Returns:
            True if entity exists, False otherwise
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get the total count of entities.

        Returns:
            Total number of entities
        """
        pass


# Alias for API consistency
DuplicateError = AlreadyExistsError