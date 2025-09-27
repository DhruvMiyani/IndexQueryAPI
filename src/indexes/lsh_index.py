"""
Locality-Sensitive Hashing (LSH) index implementation.

Approximate nearest neighbor search for high-dimensional data.
Build time: O(n)
Query time: Sub-linear (depends on hash parameters)
Space complexity: O(n + L * k * d) where L is tables, k is hash functions per table
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
import numpy as np

from .base import BaseIndex, SearchResult


class LSHIndex(BaseIndex):
    """
    LSH index for approximate vector similarity search.

    Uses random hyperplane projections to create hash signatures.
    Good for high-dimensional data where exact methods become inefficient.
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 10,
        num_hyperplanes: int = 16
    ):
        """Initialize LSH index with hyperplane parameters."""
        super().__init__(dimension)
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes

        # Storage: hash_tables[table_id][hash_key] = list of (chunk_id, vector)
        self.hash_tables: List[Dict[int, List[Tuple[UUID, List[float]]]]] = []

        # Random hyperplanes for each table
        self.hyperplanes: List["np.ndarray[Any, Any]"] = []

        # Initialize random hyperplanes
        np.random.seed(42)  # For reproducible results
        for _ in range(num_tables):
            # Create random hyperplanes (normal vectors)
            table_planes = np.random.randn(num_hyperplanes, dimension)
            # Normalize hyperplanes
            norms = np.linalg.norm(table_planes, axis=1, keepdims=True)
            table_planes = table_planes / np.maximum(norms, 1e-10)
            self.hyperplanes.append(table_planes)

        # Initialize empty hash tables
        for _ in range(num_tables):
            self.hash_tables.append({})

    def _compute_hash(self, vector: List[float], table_idx: int) -> int:
        """Compute LSH hash for a vector using specified table's hyperplanes."""
        vec_np = np.array(vector)

        # Project vector onto each hyperplane and get sign
        projections = np.dot(self.hyperplanes[table_idx], vec_np)
        hash_bits = (projections >= 0).astype(int)

        # Convert binary signature to integer hash
        hash_value = 0
        for i, bit in enumerate(hash_bits):
            hash_value |= (bit << i)

        return hash_value

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Build LSH index from vectors.

        Computes hash signatures and stores vectors in hash tables.
        """
        if not vectors:
            self.size = 0
            self.is_built = True
            return

        # Validate all vector dimensions
        for _, vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

        # Clear existing hash tables
        for table in self.hash_tables:
            table.clear()

        # Hash each vector into all tables
        for chunk_id, vector in vectors:
            for table_idx in range(self.num_tables):
                hash_key = self._compute_hash(vector, table_idx)

                if hash_key not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_key] = []

                self.hash_tables[table_idx][hash_key].append(
                    (chunk_id, vector.copy())
                )

        self.size = len(vectors)
        self.is_built = True

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a vector to the index."""
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        # Add to all hash tables
        for table_idx in range(self.num_tables):
            hash_key = self._compute_hash(vector, table_idx)

            if hash_key not in self.hash_tables[table_idx]:
                self.hash_tables[table_idx][hash_key] = []

            self.hash_tables[table_idx][hash_key].append(
                (chunk_id, vector.copy())
            )

        self.size += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k approximate nearest neighbors using LSH.

        Retrieves candidates from hash buckets and computes exact similarities.
        """
        # Validate query vector
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        if not self.is_built or k <= 0:
            return []

        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)

        # Collect candidates from all hash tables
        candidates: Set[Tuple[UUID, Tuple[float, ...]]] = set()

        for table_idx in range(self.num_tables):
            hash_key = self._compute_hash(query_vector, table_idx)

            # Get vectors in the same bucket
            if hash_key in self.hash_tables[table_idx]:
                for chunk_id, vector in self.hash_tables[table_idx][hash_key]:
                    # Use tuple to make vector hashable for set
                    candidates.add((chunk_id, tuple(vector)))

        if not candidates:
            return []

        # Compute exact similarities for candidates
        similarities = []

        for chunk_id, vector_tuple in candidates:
            vector = list(vector_tuple)
            vector_np = np.array(vector)
            vector_norm = np.linalg.norm(vector_np)

            # Calculate cosine similarity
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_np, vector_np) / (query_norm * vector_norm))

            # Clamp to [-1, 1] to handle numerical errors
            similarity = max(-1.0, min(1.0, similarity))

            similarities.append((similarity, chunk_id, vector))

        # Sort by similarity (highest first) and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similarities = similarities[:k]

        # Convert to SearchResult objects
        results = []
        for similarity, chunk_id, vector in top_similarities:
            distance = 1.0 - similarity
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        return results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        removed = False

        # Search through all hash tables and remove matching entries
        for table in self.hash_tables:
            for hash_key in list(table.keys()):
                # Filter out the chunk_id
                original_count = len(table[hash_key])
                table[hash_key] = [
                    (cid, vec) for cid, vec in table[hash_key]
                    if cid != chunk_id
                ]

                # Check if we removed anything
                if len(table[hash_key]) < original_count:
                    removed = True

                # Remove empty buckets
                if not table[hash_key]:
                    del table[hash_key]

        if removed:
            self.size -= 1

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        # Count total buckets and average bucket size
        total_buckets = sum(len(table) for table in self.hash_tables)
        non_empty_buckets = sum(
            1 for table in self.hash_tables
            for bucket in table.values() if bucket
        )

        if non_empty_buckets > 0:
            avg_bucket_size = self.size * self.num_tables / non_empty_buckets
        else:
            avg_bucket_size = 0.0

        # Calculate collision rate (how many vectors share buckets)
        total_entries = sum(
            len(bucket) for table in self.hash_tables
            for bucket in table.values()
        )
        collision_rate = total_entries / max(1, self.size * self.num_tables)

        return {
            "type": "lsh",
            "dimension": self.dimension,
            "size": self.size,
            "is_built": self.is_built,
            "num_tables": self.num_tables,
            "num_hyperplanes": self.num_hyperplanes,
            "total_buckets": total_buckets,
            "avg_bucket_size": round(avg_bucket_size, 2),
            "collision_rate": round(collision_rate, 3),
        }