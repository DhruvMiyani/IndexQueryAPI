"""
Base index interface and common types.

Defines the abstract interface for all vector indexing algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
from uuid import UUID
import numpy as np


class IndexType(Enum):
    """
    Supported index types.

    Valid values: 'linear', 'kd_tree', 'lsh', 'optimized_linear', 'improved_kd_tree', 'multiprobe_lsh', 'hnsw', 'ivf_pq'
    """
    # Original algorithms
    LINEAR = "linear"
    KD_TREE = "kd_tree"
    LSH = "lsh"

    # Optimized algorithms
    OPTIMIZED_LINEAR = "optimized_linear"
    IMPROVED_KD_TREE = "improved_kd_tree"
    MULTIPROBE_LSH = "multiprobe_lsh"

    # Advanced algorithms
    HNSW = "hnsw"
    IVF_PQ = "ivf_pq"


class Metric(str, Enum):
    """Distance/similarity metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"


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

    def __init__(self, dimension: int, metric: Metric = Metric.COSINE, normalize: bool = True):
        """
        Initialize index with vector dimension and metric.

        Args:
            dimension: The dimension of vectors in this index
            metric: Distance/similarity metric to use
            normalize: Whether to L2-normalize vectors (for cosine similarity)
        """
        self.dimension = dimension
        self.metric = metric
        self.normalize = normalize and metric == Metric.COSINE
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

    def preprocess_vector(self, vector: List[float]) -> np.ndarray:
        """
        Preprocess vector according to metric and normalization settings.

        Args:
            vector: Input vector

        Returns:
            Preprocessed numpy array
        """
        v = np.asarray(vector, dtype=np.float32)
        if self.normalize:
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
        return v

    def compute_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute similarity/distance between two vectors based on the metric.

        Args:
            v1: First vector (numpy array)
            v2: Second vector (numpy array)

        Returns:
            Similarity score (higher is more similar)
        """
        if self.metric == Metric.COSINE:
            # If normalized, cosine = dot product
            if self.normalize:
                return float(np.dot(v1, v2))
            else:
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(v1, v2) / (norm1 * norm2))
        elif self.metric == Metric.DOT_PRODUCT:
            return float(np.dot(v1, v2))
        else:  # Euclidean - return negative distance for similarity
            return -float(np.linalg.norm(v1 - v2))