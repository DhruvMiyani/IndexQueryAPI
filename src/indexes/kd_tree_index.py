"""
KD-Tree index implementation.

Efficient for exact nearest neighbor search in low-to-moderate dimensions.
Build time: O(n log n)
Query time: O(log n) best case, O(n) worst case in high dimensions
Space complexity: O(n)
"""

from typing import Any, List, Optional, Tuple, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .base import BaseIndex, SearchResult


class KDTreeNode:
    """Node in a KD-Tree."""

    def __init__(
        self,
        chunk_id: UUID,
        vector: List[float],
        split_dim: int,
        left: Optional["KDTreeNode"] = None,
        right: Optional["KDTreeNode"] = None,
    ):
        """Initialize KD-Tree node."""
        self.chunk_id = chunk_id
        self.vector = vector
        self.split_dim = split_dim
        self.left = left
        self.right = right


class KDTreeIndex(BaseIndex):
    """
    KD-Tree index for vector similarity search.

    Effective for low-to-moderate dimensional data (typically < 20 dimensions).
    Performance degrades in high dimensions due to curse of dimensionality.
    """

    def __init__(self, dimension: int):
        """Initialize empty KD-Tree index."""
        super().__init__(dimension)
        self.root: Optional[KDTreeNode] = None

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """
        Build KD-Tree from vectors.

        Recursively partitions space by cycling through dimensions.
        """
        if not vectors:
            self.root = None
            self.size = 0
            self.is_built = True
            return

        # Validate all vector dimensions
        for _, vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

        # Build tree recursively
        self.root = self._build_recursive(vectors.copy(), depth=0)
        self.size = len(vectors)
        self.is_built = True

    def _build_recursive(
        self, vectors: List[Tuple[UUID, List[float]]], depth: int
    ) -> Optional[KDTreeNode]:
        """
        Recursively build KD-Tree.

        Args:
            vectors: List of (chunk_id, vector) tuples
            depth: Current depth in tree (determines split dimension)

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
        node = KDTreeNode(
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
        """Add a vector to the index."""
        # Validate vector dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        if self.root is None:
            # Create root node
            self.root = KDTreeNode(chunk_id, vector.copy(), 0)
            self.size = 1
            return

        # Traverse tree to find insertion point
        current = self.root
        depth = 0

        while True:
            split_dim = depth % self.dimension

            if vector[split_dim] <= current.vector[split_dim]:
                # Go left
                if current.left is None:
                    current.left = KDTreeNode(chunk_id, vector.copy(), split_dim)
                    self.size += 1
                    break
                else:
                    current = current.left
            else:
                # Go right
                if current.right is None:
                    current.right = KDTreeNode(chunk_id, vector.copy(), split_dim)
                    self.size += 1
                    break
                else:
                    current = current.right

            depth += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k nearest neighbors using KD-Tree.

        Uses backtracking to ensure exact nearest neighbors are found.

        Note: KD-Tree pruning uses Euclidean distance (natural for spatial trees)
        but returns cosine similarity scores. This trade-off is acceptable since
        both metrics preserve neighborhood structure in most cases, though
        Euclidean pruning doesn't perfectly align with cosine similarity ranking.
        """
        import numpy as np
        import heapq

        # Validate query vector
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )

        if not self.root or k <= 0:
            return []

        query_np = np.array(query_vector)

        # Use a max-heap to maintain k best candidates (negate distance for max-heap)
        best_candidates: List[Tuple[float, UUID, List[float]]] = []

        def distance_squared(v1: "NDArray[Any]", v2: "NDArray[Any]") -> float:
            """Calculate squared Euclidean distance."""
            diff = v1 - v2
            return float(np.dot(diff, diff))

        def search_recursive(node: Optional[KDTreeNode]) -> None:
            """Recursively search KD-Tree with backtracking."""
            if node is None:
                return

            # Calculate distance to current node
            node_vector = np.array(node.vector)
            dist_sq = distance_squared(query_np, node_vector)

            # Add to candidates if we have room or this is closer
            if len(best_candidates) < k:
                # Use negative distance for max-heap behavior
                heapq.heappush(best_candidates, (-dist_sq, node.chunk_id, node.vector.copy()))
            elif dist_sq < -best_candidates[0][0]:  # Compare with worst candidate
                heapq.heapreplace(best_candidates, (-dist_sq, node.chunk_id, node.vector.copy()))

            # Determine which subtree to visit first
            split_dim = node.split_dim
            if query_vector[split_dim] <= node.vector[split_dim]:
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
            # (if the splitting hyperplane could contain closer points)
            if len(best_candidates) < k:
                # Still need more candidates
                search_recursive(second_child)
            else:
                # Check if other subtree could have closer points
                plane_dist_sq = (query_vector[split_dim] - node.vector[split_dim]) ** 2
                worst_dist_sq = -best_candidates[0][0]  # Convert back from negative

                if plane_dist_sq < worst_dist_sq:
                    search_recursive(second_child)

        # Start recursive search
        search_recursive(self.root)

        # Convert results to SearchResult objects with cosine similarity
        results: List[SearchResult] = []
        query_norm = np.linalg.norm(query_np)

        for _, chunk_id, vector in best_candidates:
            vector_np = np.array(vector)
            vector_norm = np.linalg.norm(vector_np)

            # Calculate cosine similarity
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query_np, vector_np) / (query_norm * vector_norm))

            # Clamp to [-1, 1] to handle numerical errors
            similarity = max(-1.0, min(1.0, similarity))
            distance = 1.0 - similarity

            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        # Note: KD-Tree removal is complex. For simplicity, we mark as removed
        # and rebuild periodically or when removal rate gets high.
        # A full implementation would require tree rebalancing.

        def find_and_remove(node: Optional[KDTreeNode], target_id: UUID) -> Optional[KDTreeNode]:
            """Find and remove node with target_id. Returns new subtree root."""
            if node is None:
                return None

            if node.chunk_id == target_id:
                # Found the node to remove
                if node.left is None and node.right is None:
                    # Leaf node - simply remove
                    return None
                elif node.left is None:
                    # Only right child
                    return node.right
                elif node.right is None:
                    # Only left child
                    return node.left
                else:
                    # Both children exist - complex case
                    # For simplicity, find minimum in right subtree and replace
                    min_node = self._find_min_in_subtree(node.right, node.split_dim)

                    # Replace current node's data with min node's data
                    node.chunk_id = min_node.chunk_id
                    node.vector = min_node.vector.copy()

                    # Remove the min node from right subtree
                    node.right = self._remove_node(node.right, min_node.chunk_id)

                return node
            else:
                # Recursively search in children
                node.left = find_and_remove(node.left, target_id)
                node.right = find_and_remove(node.right, target_id)
                return node

        original_size = self.size
        self.root = find_and_remove(self.root, chunk_id)

        # Check if size decreased (node was found and removed)
        new_size = self._count_nodes(self.root)
        if new_size < original_size:
            self.size = new_size
            return True

        return False

    def _find_min_in_subtree(self, node: Optional[KDTreeNode], dim: int) -> KDTreeNode:
        """Find node with minimum value in given dimension."""
        if node is None:
            raise ValueError("Cannot find minimum in empty subtree")

        if node.split_dim == dim:
            # If the current split dimension matches, minimum is in left subtree
            if node.left is None:
                return node
            return self._find_min_in_subtree(node.left, dim)
        else:
            # Need to check both subtrees
            candidates = [node]
            if node.left:
                candidates.append(self._find_min_in_subtree(node.left, dim))
            if node.right:
                candidates.append(self._find_min_in_subtree(node.right, dim))

            return min(candidates, key=lambda n: n.vector[dim])

    def _remove_node(self, node: Optional[KDTreeNode], target_id: UUID) -> Optional[KDTreeNode]:
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

    def _count_nodes(self, node: Optional[KDTreeNode]) -> int:
        """Count nodes in subtree."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "type": "kd_tree",
            "dimension": self.dimension,
            "size": self.size,
            "is_built": self.is_built,
            "tree_depth": self._get_tree_depth(),
            "recommended_max_dim": 20,
        }

    def _get_tree_depth(self) -> int:
        """Calculate tree depth."""
        def depth_recursive(node: Optional[KDTreeNode]) -> int:
            if node is None:
                return 0
            left_depth = depth_recursive(node.left)
            right_depth = depth_recursive(node.right)
            return 1 + max(left_depth, right_depth)

        return depth_recursive(self.root)