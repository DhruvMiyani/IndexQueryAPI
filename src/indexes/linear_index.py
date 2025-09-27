"""
Linear (brute-force) index implementation.

Performs exact nearest neighbor search by comparing query with all vectors.
Time complexity: O(n*d) per query
Space complexity: O(n*d)
"""

from typing import List, Tuple
from uuid import UUID

from .base import BaseIndex, SearchResult


class LinearIndex(BaseIndex):
    """
    Linear search index (brute-force).

    Guarantees exact nearest neighbors but with O(n) search time.
    Best for small datasets or when exact results are critical.
    """

    def __init__(self, dimension: int):
        """Initialize empty linear index."""
        super().__init__(dimension)
        self.vectors: List[Tuple[UUID, List[float]]] = []

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Build index from vectors.

        For linear index, just store the vectors.
        """
        if not vectors:
            self.vectors = []
            self.size = 0
            self.is_built = True
            return

        # Validate all vector dimensions
        for chunk_id, vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

        # Store vectors
        self.vectors = vectors.copy()
        self.size = len(vectors)
        self.is_built = True

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a vector to the index."""
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        # Check if chunk_id already exists
        for i, (existing_id, _) in enumerate(self.vectors):
            if existing_id == chunk_id:
                # Update existing vector
                self.vectors[i] = (chunk_id, vector.copy())
                return

        # Add new vector
        self.vectors.append((chunk_id, vector.copy()))
        self.size += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k nearest neighbors.

        Computes similarity with all vectors and returns top k.
        """
        import numpy as np

        # Validate query vector
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        if not self.vectors:
            return []

        if k <= 0:
            return []

        # Compute cosine similarity with all vectors
        results = []
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)

        if query_norm == 0:
            # Handle zero vector case - all similarities will be 0
            for chunk_id, _ in self.vectors:
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=0.0,
                    distance=1.0
                ))
        else:
            for chunk_id, vector in self.vectors:
                vector_np = np.array(vector)
                vector_norm = np.linalg.norm(vector_np)

                if vector_norm == 0:
                    # Handle zero vector in dataset
                    similarity = 0.0
                else:
                    # Cosine similarity: dot(a,b) / (norm(a) * norm(b))
                    similarity = float(np.dot(query_np, vector_np) / (query_norm * vector_norm))

                # Clamp to [-1, 1] to handle numerical errors
                similarity = max(-1.0, min(1.0, similarity))
                distance = 1.0 - similarity  # Convert similarity to distance

                results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=similarity,
                    distance=distance
                ))

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top k results
        return results[:k]

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        # Find vector by chunk_id
        for i, (existing_id, _) in enumerate(self.vectors):
            if existing_id == chunk_id:
                # Remove from list
                self.vectors.pop(i)
                self.size -= 1
                return True

        # Vector not found
        return False

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "type": "linear",
            "dimension": self.dimension,
            "size": self.size,
            "is_built": self.is_built,
            "memory_usage_estimate": self.dimension * self.size * 4,  # bytes
        }