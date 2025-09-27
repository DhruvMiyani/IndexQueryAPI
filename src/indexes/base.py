"""
Base index interface and common types.

Defines the abstract interface for all vector indexing algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from uuid import UUID


class IndexType(Enum):
    """Supported index types."""
    LINEAR = "linear"
    KD_TREE = "kd_tree"
    LSH = "lsh"


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    chunk_id: UUID
    score: float
    distance: float


class BaseIndex(ABC):
    """
    Abstract base class for vector indexes.

    All index implementations must inherit from this class.
    """

    def __init__(self, dimension: int):
        """
        Initialize index with vector dimension.

        Args:
            dimension: The dimension of vectors in this index
        """
        self.dimension = dimension
        self.is_built = False
        self.size = 0

    @abstractmethod
    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Build the index from a list of vectors.

        Args:
            vectors: List of (chunk_id, embedding_vector) tuples
        """
        pass

    @abstractmethod
    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """
        Add a single vector to the index.

        Args:
            chunk_id: The ID of the chunk
            vector: The embedding vector
        """
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k nearest neighbors.

        Args:
            query_vector: The query vector
            k: Number of neighbors to return

        Returns:
            List of SearchResult objects sorted by similarity
        """
        pass

    @abstractmethod
    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Args:
            chunk_id: The ID of the chunk to remove

        Returns:
            True if removed, False if not found
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        pass

    def validate_vector(self, vector: List[float]) -> None:
        """
        Validate vector dimension.

        Args:
            vector: Vector to validate

        Raises:
            ValueError: If vector dimension doesn't match index dimension
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} doesn't match "
                f"index dimension {self.dimension}"
            )

    def is_empty(self) -> bool:
        """Check if index is empty."""
        return self.size == 0