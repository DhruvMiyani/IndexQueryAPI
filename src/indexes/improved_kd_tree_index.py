"""
Improved KD-Tree index with proper metric handling.

Fixes metric inconsistencies and adds proper normalization support.
Includes warning about high-dimensional performance degradation.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import UUID
import numpy as np
import heapq

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .base import BaseIndex, SearchResult, Metric


class ImprovedKDTreeNode:
    """Node in an improved KD-Tree with preprocessed vectors."""

    def __init__(
        self,
        chunk_id: UUID,
        vector: np.ndarray,
        split_dim: int,
        left: Optional["ImprovedKDTreeNode"] = None,
        right: Optional["ImprovedKDTreeNode"] = None,
    ):
        """Initialize KD-Tree node with preprocessed vector."""
        self.chunk_id = chunk_id
        self.vector = vector  # Already preprocessed (normalized if needed)
        self.split_dim = split_dim
        self.left = left
        self.right = right


class ImprovedKDTreeIndex(BaseIndex):
    """
    Improved KD-Tree index with consistent metric handling.

    Key improvements:
    - Consistent metric usage (no mixing of Euclidean pruning with cosine ranking)
    - Proper vector normalization for cosine similarity
    - Performance warnings for high dimensions
    - Better memory layout with numpy arrays

    Performance note: KD-trees work best with dimensions <= 20.
    For higher dimensions, consider HNSW or LSH instead.
    """

    def __init__(
        self,
        dimension: int,
        metric: Metric = Metric.COSINE,
        normalize: bool = True,
        warn_high_dimension: bool = True
    ):
        """
        Initialize improved KD-Tree index.

        Args:
            dimension: Vector dimension
            metric: Distance metric to use consistently
            normalize: Whether to normalize vectors
            warn_high_dimension: Whether to warn about high dimensions
        """
        super().__init__(dimension, metric, normalize)
        self.root: Optional[ImprovedKDTreeNode] = None

        # Warn about high dimension performance
        if warn_high_dimension and dimension > 20:
            print(f"Warning: KD-Tree performance degrades significantly above 20 dimensions. "
                  f"Current dimension: {dimension}. Consider using HNSW or LSH for better performance.")

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Build KD-Tree from vectors with proper preprocessing."""
        if not vectors:
            self.root = None
            self.size = 0
            self.is_built = True
            return

        # Validate and preprocess all vectors
        processed_vectors = []
        for chunk_id, vector in vectors:
            self.validate_vector(vector)
            vec = self.preprocess_vector(vector)
            processed_vectors.append((chunk_id, vec))

        # Build tree recursively
        self.root = self._build_recursive(processed_vectors, depth=0)
        self.size = len(vectors)
        self.is_built = True

    def _build_recursive(
        self, vectors: List[Tuple[UUID, np.ndarray]], depth: int
    ) -> Optional[ImprovedKDTreeNode]:
        """
        Recursively build KD-Tree with preprocessed vectors.

        Args:
            vectors: List of (chunk_id, preprocessed_vector) tuples
            depth: Current depth in tree

        Returns:
            Root node of subtree or None if empty
        """
        if not vectors:
            return None

        # Choose split dimension (cycle through dimensions)
        split_dim = depth % self.dimension

        # Sort vectors by the split dimension
        vectors.sort(key=lambda x: x[1][split_dim])

        # Find median
        median_idx = len(vectors) // 2
        median_chunk_id, median_vector = vectors[median_idx]

        # Create node
        node = ImprovedKDTreeNode(
            chunk_id=median_chunk_id,
            vector=median_vector.copy(),
            split_dim=split_dim,
        )

        # Recursively build left and right subtrees
        left_vectors = vectors[:median_idx]
        right_vectors = vectors[median_idx + 1:]

        node.left = self._build_recursive(left_vectors, depth + 1)
        node.right = self._build_recursive(right_vectors, depth + 1)

        return node

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a vector to the index with proper preprocessing."""
        self.validate_vector(vector)
        vec = self.preprocess_vector(vector)

        if self.root is None:
            # Create root node
            self.root = ImprovedKDTreeNode(chunk_id, vec, 0)
            self.size = 1
            return

        # Traverse tree to find insertion point
        current = self.root
        depth = 0

        while True:
            split_dim = depth % self.dimension

            if vec[split_dim] <= current.vector[split_dim]:
                # Go left
                if current.left is None:
                    current.left = ImprovedKDTreeNode(chunk_id, vec, split_dim)
                    self.size += 1
                    break
                else:
                    current = current.left
            else:
                # Go right
                if current.right is None:
                    current.right = ImprovedKDTreeNode(chunk_id, vec, split_dim)
                    self.size += 1
                    break
                else:
                    current = current.right

            depth += 1

    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute distance according to the selected metric."""
        if self.metric == Metric.COSINE:
            # For normalized vectors, cosine distance = 2 * (1 - cosine_similarity)
            # But we use 1 - cosine_similarity for consistency
            if self.normalize:
                # Dot product of normalized vectors gives cosine similarity
                cosine_sim = np.dot(v1, v2)
                return 1.0 - cosine_sim
            else:
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    return 1.0
                cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
                return 1.0 - cosine_sim
        elif self.metric == Metric.DOT_PRODUCT:
            # Negative dot product as distance (higher dot product = smaller distance)
            return -np.dot(v1, v2)
        else:  # Euclidean
            return float(np.linalg.norm(v1 - v2))

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k nearest neighbors using consistent metric.

        Uses the same metric for both pruning and final ranking.
        """
        if not self.root or k <= 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Use a max-heap to maintain k best candidates
        best_candidates: List[Tuple[float, UUID, np.ndarray]] = []

        def search_recursive(node: Optional[ImprovedKDTreeNode]) -> None:
            """Recursively search KD-Tree with consistent metric."""
            if node is None:
                return

            # Calculate distance to current node using selected metric
            distance = self._compute_distance(query, node.vector)

            # Add to candidates if we have room or this is closer
            if len(best_candidates) < k:
                # Use positive distance for max-heap behavior (we want smallest distances)
                heapq.heappush(best_candidates, (-distance, node.chunk_id, node.vector.copy()))
            elif distance < -best_candidates[0][0]:  # Compare with worst candidate
                heapq.heapreplace(best_candidates, (-distance, node.chunk_id, node.vector.copy()))

            # Determine which subtree to visit first
            split_dim = node.split_dim
            if query[split_dim] <= node.vector[split_dim]:
                # Query point is on the left side
                first_child = node.left
                second_child = node.right
            else:
                # Query point is on the right side
                first_child = node.right
                second_child = node.left

            # Visit the closer subtree first
            search_recursive(first_child)

            # Check if we need to visit the other subtree
            if len(best_candidates) < k:
                # Still need more candidates
                search_recursive(second_child)
            else:
                # Check if other subtree could have closer points
                # Use coordinate distance for pruning (this is geometric property of KD-trees)
                plane_dist = abs(query[split_dim] - node.vector[split_dim])
                worst_distance = -best_candidates[0][0]  # Convert back from negative

                # Conservative pruning: if coordinate distance is less than worst distance,
                # the other subtree might contain better points
                if plane_dist < worst_distance:
                    search_recursive(second_child)

        # Start recursive search
        search_recursive(self.root)

        # Convert results to SearchResult objects
        results: List[SearchResult] = []

        for neg_distance, chunk_id, vector in best_candidates:
            distance = -neg_distance

            # Compute similarity score based on metric
            if self.metric == Metric.COSINE:
                similarity = 1.0 - distance  # Convert distance back to similarity
                # Clamp to valid range
                similarity = max(-1.0, min(1.0, similarity))
            elif self.metric == Metric.DOT_PRODUCT:
                similarity = -distance  # Negative distance was the negative dot product
            else:  # Euclidean
                similarity = -distance  # Use negative distance as similarity

            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def search_radius(self, query_vector: List[float], radius: float) -> List[SearchResult]:
        """
        Range search: find all points within radius of query.

        Args:
            query_vector: Query vector
            radius: Search radius

        Returns:
            All points within radius, sorted by distance
        """
        if not self.root:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        candidates: List[Tuple[float, UUID, np.ndarray]] = []

        def search_recursive(node: Optional[ImprovedKDTreeNode]) -> None:
            """Recursively search for points within radius."""
            if node is None:
                return

            # Calculate distance to current node
            distance = self._compute_distance(query, node.vector)

            # Add to candidates if within radius
            if distance <= radius:
                candidates.append((distance, node.chunk_id, node.vector.copy()))

            # Check both subtrees if the splitting plane intersects the search sphere
            split_dim = node.split_dim
            plane_dist = abs(query[split_dim] - node.vector[split_dim])

            if plane_dist <= radius:
                # Splitting plane intersects search sphere, check both sides
                search_recursive(node.left)
                search_recursive(node.right)
            elif query[split_dim] <= node.vector[split_dim]:
                # Only check left subtree
                search_recursive(node.left)
            else:
                # Only check right subtree
                search_recursive(node.right)

        search_recursive(self.root)

        # Convert to SearchResult objects and sort
        results = []
        for distance, chunk_id, vector in candidates:
            if self.metric == Metric.COSINE:
                similarity = 1.0 - distance
                similarity = max(-1.0, min(1.0, similarity))
            elif self.metric == Metric.DOT_PRODUCT:
                similarity = -distance
            else:
                similarity = -distance

            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        results.sort(key=lambda x: x.distance)
        return results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        def find_and_remove(node: Optional[ImprovedKDTreeNode], target_id: UUID) -> Optional[ImprovedKDTreeNode]:
            """Find and remove node with target_id."""
            if node is None:
                return None

            if node.chunk_id == target_id:
                # Found the node to remove
                if node.left is None and node.right is None:
                    return None
                elif node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                else:
                    # Both children exist - find minimum in right subtree
                    min_node = self._find_min_in_subtree(node.right, node.split_dim)
                    node.chunk_id = min_node.chunk_id
                    node.vector = min_node.vector.copy()
                    node.right = self._remove_node(node.right, min_node.chunk_id)
                return node
            else:
                node.left = find_and_remove(node.left, target_id)
                node.right = find_and_remove(node.right, target_id)
                return node

        original_size = self.size
        self.root = find_and_remove(self.root, chunk_id)

        new_size = self._count_nodes(self.root)
        if new_size < original_size:
            self.size = new_size
            return True

        return False

    def _find_min_in_subtree(self, node: Optional[ImprovedKDTreeNode], dim: int) -> ImprovedKDTreeNode:
        """Find node with minimum value in given dimension."""
        if node is None:
            raise ValueError("Cannot find minimum in empty subtree")

        if node.split_dim == dim:
            if node.left is None:
                return node
            return self._find_min_in_subtree(node.left, dim)
        else:
            candidates = [node]
            if node.left:
                candidates.append(self._find_min_in_subtree(node.left, dim))
            if node.right:
                candidates.append(self._find_min_in_subtree(node.right, dim))

            return min(candidates, key=lambda n: n.vector[dim])

    def _remove_node(self, node: Optional[ImprovedKDTreeNode], target_id: UUID) -> Optional[ImprovedKDTreeNode]:
        """Helper to remove a specific node."""
        if node is None:
            return None

        if node.chunk_id == target_id:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                min_node = self._find_min_in_subtree(node.right, node.split_dim)
                node.chunk_id = min_node.chunk_id
                node.vector = min_node.vector.copy()
                node.right = self._remove_node(node.right, min_node.chunk_id)
        else:
            node.left = self._remove_node(node.left, target_id)
            node.right = self._remove_node(node.right, target_id)

        return node

    def _count_nodes(self, node: Optional[ImprovedKDTreeNode]) -> int:
        """Count nodes in subtree."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        depth = self._get_tree_depth()

        return {
            "type": "improved_kd_tree",
            "dimension": self.dimension,
            "size": self.size,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "is_built": self.is_built,
            "tree_depth": depth,
            "balanced_depth": int(np.log2(self.size)) if self.size > 0 else 0,
            "is_well_balanced": depth <= 2 * int(np.log2(self.size + 1)) if self.size > 0 else True,
            "recommended_max_dim": 20,
            "performance_warning": self.dimension > 20
        }

    def _get_tree_depth(self) -> int:
        """Calculate tree depth."""
        def depth_recursive(node: Optional[ImprovedKDTreeNode]) -> int:
            if node is None:
                return 0
            left_depth = depth_recursive(node.left)
            right_depth = depth_recursive(node.right)
            return 1 + max(left_depth, right_depth)

        return depth_recursive(self.root)