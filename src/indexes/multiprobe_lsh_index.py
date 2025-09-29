"""
Multi-probe LSH index implementation.

Enhanced LSH with multi-probe technique for better recall.
Also explores nearby hash buckets by flipping uncertain bits.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
import numpy as np
import heapq

from .base import BaseIndex, SearchResult, Metric


class MultiProbeLSHIndex(BaseIndex):
    """
    Multi-probe LSH index for approximate similarity search.

    Improves recall over standard LSH by probing multiple nearby buckets
    based on projection magnitudes. Flips bits that have low absolute
    projection values (uncertain bits) to explore neighboring buckets.
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 8,
        num_hyperplanes: int = 32,
        max_probes: int = 8,
        candidate_limit: int = 1000,
        seed: int = 42,
        metric: Metric = Metric.COSINE,
        normalize: bool = True
    ):
        """
        Initialize multi-probe LSH index.

        Args:
            dimension: Vector dimension
            num_tables: Number of hash tables
            num_hyperplanes: Number of hyperplanes per table (hash bits)
            max_probes: Maximum number of buckets to probe per table
            candidate_limit: Maximum candidates to collect before scoring
            seed: Random seed for reproducibility
            metric: Distance metric
            normalize: Whether to normalize vectors
        """
        super().__init__(dimension, metric, normalize)

        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self.max_probes = max_probes
        self.candidate_limit = candidate_limit
        self.seed = seed

        # Initialize random number generator
        self.rng = np.random.RandomState(seed)

        # Storage: hash_tables[table_id][hash_key] = list of (chunk_id, vector)
        self.hash_tables: List[Dict[int, List[Tuple[UUID, np.ndarray]]]] = []

        # Random hyperplanes for each table
        self.hyperplanes: List[np.ndarray] = []

        # Initialize random hyperplanes
        for _ in range(num_tables):
            # Create random hyperplanes (normal vectors)
            table_planes = self.rng.randn(num_hyperplanes, dimension).astype(np.float32)
            # Normalize hyperplanes for cosine distance
            norms = np.linalg.norm(table_planes, axis=1, keepdims=True)
            table_planes = table_planes / np.maximum(norms, 1e-10)
            self.hyperplanes.append(table_planes)

        # Initialize empty hash tables
        for _ in range(num_tables):
            self.hash_tables.append({})

    def _compute_hash_with_projections(self, vector: np.ndarray, table_idx: int) -> Tuple[int, np.ndarray]:
        """
        Compute LSH hash and return both hash and projection magnitudes.

        Args:
            vector: Input vector as numpy array
            table_idx: Hash table index

        Returns:
            Tuple of (hash_value, projections)
        """
        # Project vector onto each hyperplane
        projections = np.dot(self.hyperplanes[table_idx], vector)

        # Get sign bits
        hash_bits = (projections >= 0).astype(np.int32)

        # Convert binary signature to integer hash
        hash_value = 0
        for i in range(len(hash_bits)):
            hash_value |= (int(hash_bits[i]) << i)

        return hash_value, projections

    def _generate_probe_hashes(self, projections: np.ndarray, base_hash: int) -> List[int]:
        """
        Generate additional hash keys by flipping uncertain bits.

        Args:
            projections: Projection magnitudes for each hyperplane
            base_hash: Original hash value

        Returns:
            List of probe hash values (including original)
        """
        if self.max_probes <= 1:
            return [base_hash]

        # Sort bits by absolute projection magnitude (ascending)
        # Bits with smaller |projection| are more uncertain and good to flip
        abs_projections = np.abs(projections)
        uncertain_bits = np.argsort(abs_projections)

        probe_hashes = [base_hash]

        # Generate probes by flipping combinations of uncertain bits
        for num_flips in range(1, min(self.max_probes, len(uncertain_bits) + 1)):
            if num_flips > 16:  # Avoid exponential explosion
                break

            # Flip the most uncertain bits
            flip_mask = 0
            for i in range(min(num_flips, len(uncertain_bits))):
                flip_mask |= (1 << int(uncertain_bits[i]))

            probe_hash = base_hash ^ flip_mask
            probe_hashes.append(probe_hash)

            if len(probe_hashes) >= self.max_probes:
                break

        return probe_hashes

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Build multi-probe LSH index from vectors."""
        if not vectors:
            self.size = 0
            self.is_built = True
            return

        # Validate and preprocess vectors
        processed_vectors = []
        for chunk_id, vector in vectors:
            self.validate_vector(vector)
            vec = self.preprocess_vector(vector)
            processed_vectors.append((chunk_id, vec))

        # Clear existing hash tables
        for table in self.hash_tables:
            table.clear()

        # Hash each vector into all tables
        for chunk_id, vector in processed_vectors:
            for table_idx in range(self.num_tables):
                hash_key, _ = self._compute_hash_with_projections(vector, table_idx)

                if hash_key not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_key] = []

                self.hash_tables[table_idx][hash_key].append((chunk_id, vector))

        self.size = len(vectors)
        self.is_built = True

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a vector to the index."""
        self.validate_vector(vector)
        vec = self.preprocess_vector(vector)

        # Add to all hash tables
        for table_idx in range(self.num_tables):
            hash_key, _ = self._compute_hash_with_projections(vec, table_idx)

            if hash_key not in self.hash_tables[table_idx]:
                self.hash_tables[table_idx][hash_key] = []

            # Check if chunk already exists and update
            found = False
            for i, (existing_id, _) in enumerate(self.hash_tables[table_idx][hash_key]):
                if existing_id == chunk_id:
                    self.hash_tables[table_idx][hash_key][i] = (chunk_id, vec)
                    found = True
                    break

            if not found:
                self.hash_tables[table_idx][hash_key].append((chunk_id, vec))

        if chunk_id not in [cid for table in self.hash_tables for bucket in table.values() for cid, _ in bucket]:
            self.size += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Multi-probe search for k approximate nearest neighbors.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return

        Returns:
            Top k search results sorted by similarity
        """
        if not self.is_built or k <= 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Collect candidates from all hash tables with multi-probe
        candidate_map: Dict[UUID, np.ndarray] = {}

        for table_idx in range(self.num_tables):
            base_hash, projections = self._compute_hash_with_projections(query, table_idx)
            probe_hashes = self._generate_probe_hashes(projections, base_hash)

            for hash_key in probe_hashes:
                if hash_key in self.hash_tables[table_idx]:
                    for chunk_id, vector in self.hash_tables[table_idx][hash_key]:
                        if chunk_id not in candidate_map:
                            candidate_map[chunk_id] = vector

                        # Early termination if we have too many candidates
                        if len(candidate_map) >= self.candidate_limit:
                            break

                if len(candidate_map) >= self.candidate_limit:
                    break

            if len(candidate_map) >= self.candidate_limit:
                break

        if not candidate_map:
            return []

        # Compute exact similarities for candidates
        similarities = []
        for chunk_id, vector in candidate_map.items():
            similarity = self.compute_similarity(query, vector)
            similarities.append((similarity, chunk_id, vector))

        # Sort by similarity (highest first) and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similarities = similarities[:k]

        # Convert to SearchResult objects
        results = []
        for similarity, chunk_id, vector in top_similarities:
            if self.metric == Metric.COSINE:
                distance = 1.0 - similarity
            elif self.metric == Metric.DOT_PRODUCT:
                distance = -similarity
            else:  # Euclidean
                distance = -similarity

            results.append(SearchResult(
                chunk_id=chunk_id,
                score=similarity,
                distance=distance
            ))

        return results

    def search_with_recall_tuning(
        self,
        query_vector: List[float],
        k: int,
        target_recall: float = 0.9
    ) -> List[SearchResult]:
        """
        Search with automatic probe count adjustment for target recall.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return
            target_recall: Desired recall (between 0 and 1)

        Returns:
            Search results with dynamically adjusted probe count
        """
        # Simple heuristic: increase probes based on target recall
        if target_recall >= 0.95:
            original_probes = self.max_probes
            self.max_probes = min(16, self.num_hyperplanes // 2)
        elif target_recall >= 0.8:
            original_probes = self.max_probes
            self.max_probes = min(12, original_probes + 4)
        else:
            original_probes = self.max_probes

        try:
            results = self.search(query_vector, k)
        finally:
            self.max_probes = original_probes

        return results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        removed = False

        # Search through all hash tables and remove matching entries
        for table in self.hash_tables:
            for hash_key in list(table.keys()):
                original_count = len(table[hash_key])
                table[hash_key] = [
                    (cid, vec) for cid, vec in table[hash_key]
                    if cid != chunk_id
                ]

                if len(table[hash_key]) < original_count:
                    removed = True

                # Remove empty buckets
                if not table[hash_key]:
                    del table[hash_key]

        if removed:
            self.size -= 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
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

        # Calculate hash distribution statistics
        bucket_sizes = []
        for table in self.hash_tables:
            for bucket in table.values():
                bucket_sizes.append(len(bucket))

        if bucket_sizes:
            max_bucket_size = max(bucket_sizes)
            min_bucket_size = min(bucket_sizes)
            avg_entries_per_table = sum(bucket_sizes) / len(self.hash_tables)
        else:
            max_bucket_size = min_bucket_size = avg_entries_per_table = 0

        # Estimate memory usage
        vector_memory = self.size * self.dimension * 4  # float32
        hyperplane_memory = sum(hp.nbytes for hp in self.hyperplanes)
        bucket_overhead = total_buckets * 64  # rough estimate for dict overhead

        return {
            "type": "multiprobe_lsh",
            "dimension": self.dimension,
            "size": self.size,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "is_built": self.is_built,
            "num_tables": self.num_tables,
            "num_hyperplanes": self.num_hyperplanes,
            "max_probes": self.max_probes,
            "candidate_limit": self.candidate_limit,
            "total_buckets": total_buckets,
            "avg_bucket_size": round(avg_bucket_size, 2),
            "max_bucket_size": max_bucket_size,
            "min_bucket_size": min_bucket_size,
            "avg_entries_per_table": round(avg_entries_per_table, 2),
            "memory_usage_bytes": vector_memory + hyperplane_memory + bucket_overhead,
            "seed": self.seed
        }