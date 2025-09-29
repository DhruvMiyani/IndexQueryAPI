"""
Optimized linear index with vectorization.

Uses numpy matrix operations for fast brute-force search.
Maintains contiguous memory layout for cache efficiency.
"""

from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
import numpy as np

from .base import BaseIndex, SearchResult, Metric


class OptimizedLinearIndex(BaseIndex):
    """
    Vectorized linear search index.

    Uses numpy matrix operations for efficient exact nearest neighbor search.
    Maintains data in contiguous numpy arrays for optimal performance.
    """

    def __init__(self, dimension: int, metric: Metric = Metric.COSINE, normalize: bool = True):
        """Initialize optimized linear index."""
        super().__init__(dimension, metric, normalize)

        # Pre-allocate arrays with initial capacity
        self.initial_capacity = 1000
        self.capacity = self.initial_capacity

        # Contiguous numpy array for vectors (row-major for matrix multiply)
        self.vectors = np.empty((self.capacity, dimension), dtype=np.float32)

        # ID mappings
        self.ids: List[UUID] = []
        self.id_to_idx: Dict[UUID, int] = {}

        # Track actual size
        self.size = 0

    def _resize_if_needed(self) -> None:
        """Resize arrays if capacity is reached."""
        if self.size >= self.capacity:
            new_capacity = self.capacity * 2
            new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:self.size] = self.vectors[:self.size]
            self.vectors = new_vectors
            self.capacity = new_capacity

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Build index from vectors using batch operations."""
        if not vectors:
            self.size = 0
            self.is_built = True
            return

        n = len(vectors)

        # Ensure capacity
        if n > self.capacity:
            self.capacity = n * 2
            self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)

        # Batch validate and process
        self.ids = []
        self.id_to_idx = {}

        for i, (chunk_id, vector) in enumerate(vectors):
            self.validate_vector(vector)

            # Preprocess and store
            vec = self.preprocess_vector(vector)
            self.vectors[i] = vec
            self.ids.append(chunk_id)
            self.id_to_idx[chunk_id] = i

        self.size = n
        self.is_built = True

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add or update a vector in the index."""
        self.validate_vector(vector)
        vec = self.preprocess_vector(vector)

        if chunk_id in self.id_to_idx:
            # Update existing vector
            idx = self.id_to_idx[chunk_id]
            self.vectors[idx] = vec
        else:
            # Add new vector
            self._resize_if_needed()
            self.vectors[self.size] = vec
            self.ids.append(chunk_id)
            self.id_to_idx[chunk_id] = self.size
            self.size += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Vectorized k-NN search using matrix operations.

        For cosine similarity with normalized vectors, this becomes a simple
        matrix multiplication: scores = X @ q
        """
        if self.size == 0 or k <= 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Get the active vectors (only up to self.size)
        X = self.vectors[:self.size]

        # Compute similarities using vectorized operations
        if self.metric == Metric.COSINE:
            if self.normalize:
                # For normalized vectors, cosine = dot product
                # Use matrix multiplication for all similarities at once
                scores = X @ query  # Shape: (n,)
            else:
                # Compute norms and dot products
                dots = X @ query
                x_norms = np.linalg.norm(X, axis=1)
                q_norm = np.linalg.norm(query)

                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    scores = dots / (x_norms * q_norm + 1e-10)
                    scores = np.nan_to_num(scores, nan=0.0)

        elif self.metric == Metric.DOT_PRODUCT:
            scores = X @ query

        else:  # Euclidean
            # Compute squared Euclidean distances efficiently
            # ||x - q||^2 = ||x||^2 + ||q||^2 - 2<x,q>
            x_sqnorms = np.sum(X * X, axis=1)
            q_sqnorm = np.sum(query * query)
            dots = X @ query
            distances = x_sqnorms + q_sqnorm - 2 * dots
            scores = -np.sqrt(np.maximum(distances, 0))  # Negative for similarity

        # Use argpartition for O(n + k log k) instead of O(n log n) full sort
        k_actual = min(k, self.size)

        if k_actual == self.size:
            # Need all elements, just sort
            top_indices = np.argsort(-scores)
        else:
            # Use argpartition to find top k efficiently
            # Note: argpartition with negative scores to get largest
            partition_indices = np.argpartition(-scores, k_actual - 1)
            top_k_indices = partition_indices[:k_actual]

            # Sort only the top k
            top_k_scores = scores[top_k_indices]
            sorted_within_top_k = np.argsort(-top_k_scores)
            top_indices = top_k_indices[sorted_within_top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])

            if self.metric == Metric.COSINE:
                distance = 1.0 - score
            elif self.metric == Metric.DOT_PRODUCT:
                distance = -score
            else:  # Euclidean
                distance = -score

            results.append(SearchResult(
                chunk_id=self.ids[idx],
                score=score,
                distance=distance
            ))

        return results

    def search_with_filter(
        self,
        query_vector: List[float],
        k: int,
        filter_mask: Optional[np.ndarray] = None
    ) -> List[SearchResult]:
        """
        Search with an optional boolean filter mask.

        Args:
            query_vector: Query vector
            k: Number of results
            filter_mask: Boolean array where True indicates vectors to consider

        Returns:
            Top k filtered results
        """
        if filter_mask is None:
            return self.search(query_vector, k)

        if self.size == 0 or k <= 0:
            return []

        # Apply filter
        filtered_indices = np.where(filter_mask[:self.size])[0]
        if len(filtered_indices) == 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Get filtered vectors
        X_filtered = self.vectors[filtered_indices]

        # Compute similarities for filtered vectors only
        if self.metric == Metric.COSINE and self.normalize:
            scores = X_filtered @ query
        else:
            # Similar logic as in search() but for filtered vectors
            if self.metric == Metric.COSINE:
                dots = X_filtered @ query
                x_norms = np.linalg.norm(X_filtered, axis=1)
                q_norm = np.linalg.norm(query)
                with np.errstate(divide='ignore', invalid='ignore'):
                    scores = dots / (x_norms * q_norm + 1e-10)
                    scores = np.nan_to_num(scores, nan=0.0)
            elif self.metric == Metric.DOT_PRODUCT:
                scores = X_filtered @ query
            else:  # Euclidean
                x_sqnorms = np.sum(X_filtered * X_filtered, axis=1)
                q_sqnorm = np.sum(query * query)
                dots = X_filtered @ query
                distances = x_sqnorms + q_sqnorm - 2 * dots
                scores = -np.sqrt(np.maximum(distances, 0))

        # Get top k from filtered results
        k_actual = min(k, len(filtered_indices))
        if k_actual == len(filtered_indices):
            top_local_indices = np.argsort(-scores)
        else:
            partition_indices = np.argpartition(-scores, k_actual - 1)
            top_k_indices = partition_indices[:k_actual]
            top_k_scores = scores[top_k_indices]
            sorted_within_top_k = np.argsort(-top_k_scores)
            top_local_indices = top_k_indices[sorted_within_top_k]

        # Map back to original indices and build results
        results = []
        for local_idx in top_local_indices:
            original_idx = filtered_indices[local_idx]
            score = float(scores[local_idx])

            if self.metric == Metric.COSINE:
                distance = 1.0 - score
            elif self.metric == Metric.DOT_PRODUCT:
                distance = -score
            else:
                distance = -score

            results.append(SearchResult(
                chunk_id=self.ids[original_idx],
                score=score,
                distance=distance
            ))

        return results

    def batch_search(
        self,
        query_vectors: List[List[float]],
        k: int
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries simultaneously.

        Args:
            query_vectors: List of query vectors
            k: Number of results per query

        Returns:
            List of search results for each query
        """
        if self.size == 0 or k <= 0:
            return [[] for _ in query_vectors]

        # Preprocess all queries
        queries = np.array([
            self.preprocess_vector(q) for q in query_vectors
        ], dtype=np.float32)

        X = self.vectors[:self.size]

        # Compute all similarities at once using matrix multiplication
        # Shape: (num_queries, num_vectors)
        if self.metric == Metric.COSINE and self.normalize:
            all_scores = queries @ X.T
        else:
            # Process in batches if needed for other metrics
            all_scores = np.zeros((len(queries), self.size), dtype=np.float32)
            for i, query in enumerate(queries):
                if self.metric == Metric.COSINE:
                    dots = X @ query
                    x_norms = np.linalg.norm(X, axis=1)
                    q_norm = np.linalg.norm(query)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        all_scores[i] = dots / (x_norms * q_norm + 1e-10)
                elif self.metric == Metric.DOT_PRODUCT:
                    all_scores[i] = X @ query
                else:  # Euclidean
                    diffs = X - query
                    all_scores[i] = -np.linalg.norm(diffs, axis=1)

        # Process results for each query
        all_results = []
        for query_scores in all_scores:
            k_actual = min(k, self.size)

            if k_actual == self.size:
                top_indices = np.argsort(-query_scores)
            else:
                partition_indices = np.argpartition(-query_scores, k_actual - 1)
                top_k_indices = partition_indices[:k_actual]
                top_k_scores = query_scores[top_k_indices]
                sorted_within_top_k = np.argsort(-top_k_scores)
                top_indices = top_k_indices[sorted_within_top_k]

            results = []
            for idx in top_indices:
                score = float(query_scores[idx])

                if self.metric == Metric.COSINE:
                    distance = 1.0 - score
                elif self.metric == Metric.DOT_PRODUCT:
                    distance = -score
                else:
                    distance = -score

                results.append(SearchResult(
                    chunk_id=self.ids[idx],
                    score=score,
                    distance=distance
                ))

            all_results.append(results)

        return all_results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        if chunk_id not in self.id_to_idx:
            return False

        idx = self.id_to_idx[chunk_id]

        # Swap with last element and remove
        if idx < self.size - 1:
            # Move last element to this position
            self.vectors[idx] = self.vectors[self.size - 1]
            self.ids[idx] = self.ids[self.size - 1]

            # Update mapping for the moved element
            moved_id = self.ids[idx]
            self.id_to_idx[moved_id] = idx

        # Remove the last element (which is now the one we want to delete)
        self.ids.pop()
        del self.id_to_idx[chunk_id]
        self.size -= 1

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "type": "optimized_linear",
            "dimension": self.dimension,
            "size": self.size,
            "capacity": self.capacity,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "is_built": self.is_built,
            "memory_usage_bytes": self.vectors[:self.size].nbytes + len(self.ids) * 64,
            "vectorized": True
        }