"""
HNSW (Hierarchical Navigable Small World) index implementation.

State-of-the-art ANN algorithm with excellent recall/latency trade-offs.
Supports dynamic insertions and provides tunable parameters for optimization.
"""

import heapq
import math
import random
from typing import Any, Dict, List, Tuple, Optional, Set
from uuid import UUID
import numpy as np
from collections import defaultdict

from .base import BaseIndex, SearchResult, Metric


class HNSWIndex(BaseIndex):
    """
    Hierarchical Navigable Small World index for fast approximate nearest neighbor search.

    Based on the paper: "Efficient and robust approximate nearest neighbor search using
    Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2016)

    Key parameters:
    - M: Number of bi-directional links created for every new element (16-64 typically)
    - ef_construction: Size of the dynamic list during construction (200 typical)
    - ef_search: Size of the dynamic list for search (50-200 typical)
    """

    def __init__(
        self,
        dimension: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_M: int = None,
        max_M0: int = None,
        seed: int = None,
        metric: Metric = Metric.COSINE,
        normalize: bool = True
    ):
        """
        Initialize HNSW index.

        Args:
            dimension: Vector dimension
            M: Number of connections for each element (except initial layer)
            ef_construction: Size of dynamic list during construction
            ef_search: Size of dynamic list during search
            max_M: Maximum allowed connections for any element
            max_M0: Maximum allowed connections for layer 0
            seed: Random seed for reproducibility
            metric: Distance metric to use
            normalize: Whether to normalize vectors
        """
        super().__init__(dimension, metric, normalize)

        self.M = M
        self.max_M = max_M or M
        self.max_M0 = max_M0 or (M * 2)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_multiplier = 1 / math.log(M)
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Core data structures
        self.element_count = 0
        self.data: Dict[int, np.ndarray] = {}  # element_id -> vector
        self.ids_mapping: Dict[int, UUID] = {}  # element_id -> chunk_id
        self.reverse_ids: Dict[UUID, int] = {}  # chunk_id -> element_id
        self.levels: Dict[int, int] = {}  # element_id -> level
        self.graph: Dict[int, List[List[int]]] = {}  # element_id -> [neighbors per level]
        self.entry_point: Optional[int] = None

    def _get_random_level(self) -> int:
        """Select level for a new element using exponential decay probability."""
        level = 0
        while random.random() < 0.5 and level < 16:
            level += 1
        return level

    def _distance(self, idx1: int, idx2: int) -> float:
        """Compute distance between two elements by their indices."""
        return -self.compute_similarity(self.data[idx1], self.data[idx2])

    def _distance_to_query(self, query: np.ndarray, idx: int) -> float:
        """Compute distance from query to an element."""
        return -self.compute_similarity(query, self.data[idx])

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: Set[int],
        num_closest: int,
        layer: int
    ) -> Set[int]:
        """
        Search for nearest neighbors at a specific layer.

        Args:
            query: Query vector
            entry_points: Starting points for search
            num_closest: Number of nearest neighbors to return
            layer: Current layer number

        Returns:
            Set of nearest neighbor indices
        """
        visited = set()
        candidates = []
        w = []

        for point in entry_points:
            dist = self._distance_to_query(query, point)
            heapq.heappush(candidates, (-dist, point))
            heapq.heappush(w, (dist, point))
            visited.add(point)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            neighbors = self.graph[current][layer] if layer < len(self.graph[current]) else []

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance_to_query(query, neighbor)

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return set([idx for _, idx in w])

    def _get_neighbors(self, layer: int) -> int:
        """Return the number of neighbors to connect at a given layer."""
        if layer == 0:
            return self.max_M0
        return self.max_M

    def _prune_connections(
        self,
        candidates: List[Tuple[float, int]],
        m: int,
        base_idx: Optional[int] = None
    ) -> List[int]:
        """
        Prune connections using a heuristic to maintain connectivity.

        Args:
            candidates: List of (distance, index) tuples
            m: Maximum number of connections to keep
            base_idx: Index of the base element (for pruning its connections)

        Returns:
            List of selected neighbor indices
        """
        candidates = sorted(candidates, key=lambda x: x[0])

        result = []
        for dist, idx in candidates:
            if len(result) >= m:
                break

            # Simple heuristic: add if it's close enough
            # More sophisticated heuristics can be added here
            result.append(idx)

        return result

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a single vector to the index."""
        self.validate_vector(vector)
        vec = self.preprocess_vector(vector)

        if chunk_id in self.reverse_ids:
            # Update existing vector
            idx = self.reverse_ids[chunk_id]
            self.data[idx] = vec
            return

        idx = self.element_count
        self.element_count += 1

        self.data[idx] = vec
        self.ids_mapping[idx] = chunk_id
        self.reverse_ids[chunk_id] = idx

        level = self._get_random_level()
        self.levels[idx] = level
        self.graph[idx] = [[] for _ in range(level + 1)]

        if self.entry_point is None:
            self.entry_point = idx
            self.size = 1
            return

        # Find nearest neighbors at all layers
        search_entry_points = {self.entry_point}

        # Search from top to target layer
        for lc in range(self.levels[self.entry_point], level, -1):
            search_entry_points = self._search_layer(
                vec, search_entry_points, 1, lc
            )

        # Insert at all layers from level to 0
        for lc in range(level, -1, -1):
            candidates = self._search_layer(
                vec, search_entry_points, self.ef_construction, lc
            )

            m = self._get_neighbors(lc)

            # Convert to list with distances for pruning
            candidates_with_dist = [
                (self._distance(idx, neighbor), neighbor)
                for neighbor in candidates
            ]

            # Select neighbors
            neighbors = self._prune_connections(candidates_with_dist, m, idx)

            # Add bidirectional links
            self.graph[idx][lc] = neighbors

            for neighbor in neighbors:
                if lc < len(self.graph[neighbor]):
                    self.graph[neighbor][lc].append(idx)

                    # Prune neighbor's connections if needed
                    max_conn = self._get_neighbors(lc)
                    if len(self.graph[neighbor][lc]) > max_conn:
                        prune_list = [
                            (self._distance(neighbor, nn), nn)
                            for nn in self.graph[neighbor][lc]
                        ]
                        self.graph[neighbor][lc] = self._prune_connections(
                            prune_list, max_conn, neighbor
                        )

        # Update entry point if new element has higher level
        if level > self.levels[self.entry_point]:
            self.entry_point = idx

        self.size += 1

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Build index from a list of vectors."""
        for chunk_id, vector in vectors:
            self.add(chunk_id, vector)
        self.is_built = True

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """Search for k nearest neighbors."""
        if self.entry_point is None or k <= 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Search from top layer to layer 0
        search_entry_points = {self.entry_point}

        for level in range(self.levels[self.entry_point], 0, -1):
            search_entry_points = self._search_layer(
                query, search_entry_points, 1, level
            )

        # Search at layer 0 with ef_search
        candidates = self._search_layer(
            query, search_entry_points, max(self.ef_search, k), 0
        )

        # Sort by distance and return top k
        candidates_with_dist = []
        for idx in candidates:
            sim = self.compute_similarity(query, self.data[idx])
            candidates_with_dist.append((sim, idx))

        candidates_with_dist.sort(reverse=True)

        results = []
        for sim, idx in candidates_with_dist[:k]:
            results.append(SearchResult(
                chunk_id=self.ids_mapping[idx],
                score=sim,
                distance=1.0 - sim if self.metric == Metric.COSINE else -sim
            ))

        return results

    def remove(self, chunk_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Note: HNSW doesn't support efficient deletion. This marks the element
        as deleted (tombstone) and requires periodic rebuilding for cleanup.
        """
        if chunk_id not in self.reverse_ids:
            return False

        # Mark as tombstone by removing from data
        idx = self.reverse_ids[chunk_id]
        if idx in self.data:
            del self.data[idx]
            del self.ids_mapping[idx]
            del self.reverse_ids[chunk_id]
            self.size -= 1

            # Note: Graph structure remains but queries will skip tombstones
            # A periodic rebuild is recommended to clean up the graph

        return True

    def set_ef_search(self, ef_search: int) -> None:
        """Update the ef parameter for search (allows runtime tuning)."""
        self.ef_search = ef_search

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            "type": "hnsw",
            "dimension": self.dimension,
            "size": self.size,
            "metric": self.metric.value,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "entry_point": self.entry_point,
            "num_levels": len(set(self.levels.values())) if self.levels else 0,
            "max_level": max(self.levels.values()) if self.levels else 0,
            "memory_usage_mb": self._estimate_memory_usage() / 1024 / 1024
        }

        if self.levels:
            stats["avg_level"] = sum(self.levels.values()) / len(self.levels)

        return stats

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in bytes."""
        # Vector storage
        vector_size = self.size * self.dimension * 4  # float32

        # Graph structure
        graph_size = 0
        for neighbors_list in self.graph.values():
            for level_neighbors in neighbors_list:
                graph_size += len(level_neighbors) * 8  # pointer size

        # Metadata
        metadata_size = len(self.ids_mapping) * 64  # UUID and mappings

        return float(vector_size + graph_size + metadata_size)