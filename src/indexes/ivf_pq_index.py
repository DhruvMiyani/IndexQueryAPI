"""
IVF-PQ (Inverted File with Product Quantization) index implementation.

Combines coarse quantization (IVF) with fine quantization (PQ) for memory-efficient
approximate nearest neighbor search on large datasets.

Based on:
- "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)
- FAISS implementation patterns
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import numpy as np
from collections import defaultdict

from .base import BaseIndex, SearchResult, Metric


class ProductQuantizer:
    """Product Quantizer for vector compression."""

    def __init__(self, dimension: int, m: int = 8, nbits: int = 8, seed: int = 42):
        """
        Initialize Product Quantizer.

        Args:
            dimension: Vector dimension
            m: Number of subvectors (must divide dimension)
            nbits: Bits per subquantizer (2^nbits centroids per subspace)
            seed: Random seed for reproducibility
        """
        if dimension % m != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by m={m}")

        self.dimension = dimension
        self.m = m
        self.nbits = nbits
        self.ksub = 2 ** nbits  # centroids per subspace
        self.dsub = dimension // m  # dimension per subspace
        self.seed = seed

        # Codebooks: m subquantizers, each with ksub centroids of dsub dimensions
        self.codebooks: Optional[np.ndarray] = None  # Shape: (m, ksub, dsub)
        self.is_trained = False

        self.rng = np.random.RandomState(seed)

    def train(self, vectors: np.ndarray) -> None:
        """
        Train the quantizer on a set of vectors.

        Args:
            vectors: Training vectors, shape (n, dimension)
        """
        n, d = vectors.shape
        assert d == self.dimension

        # Split vectors into subvectors
        subvectors = vectors.reshape(n * self.m, self.dsub)

        # Train k-means for each subspace
        self.codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float32)

        for i in range(self.m):
            # Get subvectors for this subspace
            start_idx = i * n
            end_idx = (i + 1) * n
            sub_data = subvectors[start_idx:end_idx]

            # Simple k-means clustering
            centroids = self._kmeans(sub_data, self.ksub)
            self.codebooks[i] = centroids

        self.is_trained = True

    def _kmeans(self, data: np.ndarray, k: int, max_iters: int = 20) -> np.ndarray:
        """Simple k-means implementation."""
        n, d = data.shape

        # Initialize centroids randomly
        centroids = data[self.rng.choice(n, k, replace=False)]

        for _ in range(max_iters):
            # Assign points to nearest centroids
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[j] = centroids[j]  # Keep old centroid if no assignments

            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        return centroids

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors into PQ codes.

        Args:
            vectors: Input vectors, shape (n, dimension)

        Returns:
            PQ codes, shape (n, m)
        """
        if not self.is_trained:
            raise ValueError("Quantizer must be trained before encoding")

        n = vectors.shape[0]
        codes = np.zeros((n, self.m), dtype=np.uint8)

        for i in range(self.m):
            # Extract subvectors for this subspace
            start_dim = i * self.dsub
            end_dim = (i + 1) * self.dsub
            subvectors = vectors[:, start_dim:end_dim]

            # Find nearest centroid for each subvector
            centroids = self.codebooks[i]
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to approximate vectors.

        Args:
            codes: PQ codes, shape (n, m)

        Returns:
            Decoded vectors, shape (n, dimension)
        """
        n, m = codes.shape
        vectors = np.zeros((n, self.dimension), dtype=np.float32)

        for i in range(m):
            start_dim = i * self.dsub
            end_dim = (i + 1) * self.dsub
            vectors[:, start_dim:end_dim] = self.codebooks[i][codes[:, i]]

        return vectors


class IVFPQIndex(BaseIndex):
    """
    IVF-PQ index for memory-efficient large-scale vector search.

    Combines:
    - IVF (Inverted File): Coarse quantization to partition space
    - PQ (Product Quantization): Fine quantization for compression

    Good for datasets with millions+ vectors where memory is constrained.
    """

    def __init__(
        self,
        dimension: int,
        nlist: int = 4096,
        m: int = 8,
        nbits: int = 8,
        nprobe: int = 8,
        rerank_size: int = 64,
        metric: Metric = Metric.COSINE,
        normalize: bool = True,
        seed: int = 42
    ):
        """
        Initialize IVF-PQ index.

        Args:
            dimension: Vector dimension
            nlist: Number of IVF clusters (coarse quantization)
            m: Number of PQ subvectors (must divide dimension)
            nbits: Bits per PQ subquantizer
            nprobe: Number of clusters to probe during search
            rerank_size: Number of candidates to rerank with full precision
            metric: Distance metric
            normalize: Whether to normalize vectors
            seed: Random seed
        """
        super().__init__(dimension, metric, normalize)

        self.nlist = nlist
        self.nprobe = nprobe
        self.rerank_size = rerank_size
        self.seed = seed

        # Coarse quantizer (IVF centroids)
        self.coarse_centroids: Optional[np.ndarray] = None  # Shape: (nlist, dimension)

        # Product quantizer for fine quantization
        self.pq = ProductQuantizer(dimension, m, nbits, seed)

        # Inverted lists: cluster_id -> list of (chunk_id, pq_code)
        self.inverted_lists: Dict[int, List[Tuple[UUID, np.ndarray]]] = defaultdict(list)

        # Full precision vectors for reranking (optional)
        self.full_vectors: Dict[UUID, np.ndarray] = {}

        self.rng = np.random.RandomState(seed)
        self.is_trained = False

    def _train_coarse_quantizer(self, vectors: np.ndarray) -> None:
        """Train the coarse quantizer (k-means for IVF)."""
        n = vectors.shape[0]

        if n < self.nlist:
            # Not enough vectors, use all as centroids
            self.coarse_centroids = vectors.copy()
            print(f"Warning: Only {n} vectors available, using all as coarse centroids")
        else:
            # Use k-means to find coarse centroids
            self.coarse_centroids = self._kmeans(vectors, self.nlist)

    def _kmeans(self, data: np.ndarray, k: int, max_iters: int = 25) -> np.ndarray:
        """K-means clustering for coarse quantization."""
        n, d = data.shape

        # Initialize centroids using k-means++
        centroids = np.zeros((k, d), dtype=np.float32)
        centroids[0] = data[self.rng.randint(n)]

        for i in range(1, k):
            # Compute distances to nearest existing centroid
            distances = np.min(
                np.linalg.norm(data[:, np.newaxis, :] - centroids[:i], axis=2), axis=1
            )
            # Choose next centroid with probability proportional to squared distance
            probs = distances ** 2
            probs /= np.sum(probs)
            next_idx = self.rng.choice(n, p=probs)
            centroids[i] = data[next_idx]

        # Lloyd's algorithm
        for iteration in range(max_iters):
            # Assign points to nearest centroids
            distances = np.linalg.norm(
                data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
            )
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    new_centroids[j] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[j] = centroids[j]

            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break

            centroids = new_centroids

        return centroids

    def _assign_to_cluster(self, vector: np.ndarray) -> int:
        """Assign vector to nearest coarse cluster."""
        if self.coarse_centroids is None:
            raise ValueError("Coarse quantizer not trained")

        # Find nearest centroid
        distances = np.linalg.norm(self.coarse_centroids - vector, axis=1)
        return int(np.argmin(distances))

    def build(self, vectors: List[Tuple[UUID, List[float]]]) -> None:
        """Build IVF-PQ index from vectors."""
        if not vectors:
            self.size = 0
            self.is_built = True
            return

        # Preprocess vectors
        processed_vectors = []
        vector_matrix = np.zeros((len(vectors), self.dimension), dtype=np.float32)

        for i, (chunk_id, vector) in enumerate(vectors):
            self.validate_vector(vector)
            vec = self.preprocess_vector(vector)
            processed_vectors.append((chunk_id, vec))
            vector_matrix[i] = vec

        # Train coarse quantizer
        print(f"Training coarse quantizer with {len(vectors)} vectors...")
        self._train_coarse_quantizer(vector_matrix)

        # Train product quantizer
        print(f"Training product quantizer...")
        self.pq.train(vector_matrix)

        # Clear inverted lists
        self.inverted_lists.clear()
        self.full_vectors.clear()

        # Add vectors to inverted lists
        print(f"Building inverted lists...")
        for chunk_id, vector in processed_vectors:
            cluster_id = self._assign_to_cluster(vector)
            pq_code = self.pq.encode(vector.reshape(1, -1))[0]

            self.inverted_lists[cluster_id].append((chunk_id, pq_code))

            # Store some full precision vectors for reranking
            if len(self.full_vectors) < self.rerank_size * 10:
                self.full_vectors[chunk_id] = vector

        self.size = len(vectors)
        self.is_trained = True
        self.is_built = True

        print(f"IVF-PQ index built: {self.nlist} clusters, {self.size} vectors")

    def add(self, chunk_id: UUID, vector: List[float]) -> None:
        """Add a vector to the index."""
        if not self.is_trained:
            raise ValueError("Index must be built before adding individual vectors")

        self.validate_vector(vector)
        vec = self.preprocess_vector(vector)

        # Assign to cluster and encode
        cluster_id = self._assign_to_cluster(vec)
        pq_code = self.pq.encode(vec.reshape(1, -1))[0]

        # Remove if already exists
        self.remove(chunk_id)

        # Add to inverted list
        self.inverted_lists[cluster_id].append((chunk_id, pq_code))

        # Optionally store full precision for reranking
        if len(self.full_vectors) < self.rerank_size * 10:
            self.full_vectors[chunk_id] = vec

        self.size += 1

    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for k nearest neighbors using IVF-PQ.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return

        Returns:
            Top k approximate nearest neighbors
        """
        if not self.is_trained or k <= 0:
            return []

        self.validate_vector(query_vector)
        query = self.preprocess_vector(query_vector)

        # Find nearest coarse clusters to probe
        distances_to_centroids = np.linalg.norm(self.coarse_centroids - query, axis=1)
        probe_clusters = np.argpartition(distances_to_centroids, self.nprobe)[:self.nprobe]

        # Collect candidates from probed clusters
        candidates: List[Tuple[UUID, np.ndarray]] = []

        for cluster_id in probe_clusters:
            for chunk_id, pq_code in self.inverted_lists[cluster_id]:
                # Decode PQ code to approximate vector
                approx_vector = self.pq.decode(pq_code.reshape(1, -1))[0]
                candidates.append((chunk_id, approx_vector))

        if not candidates:
            return []

        # Score candidates using approximate vectors
        scored_candidates = []
        for chunk_id, approx_vector in candidates:
            similarity = self.compute_similarity(query, approx_vector)
            scored_candidates.append((similarity, chunk_id))

        # Sort and take top candidates for potential reranking
        scored_candidates.sort(reverse=True)
        top_candidates = scored_candidates[:min(self.rerank_size, len(scored_candidates))]

        # Rerank using full precision vectors if available
        final_results = []
        for similarity, chunk_id in top_candidates:
            if chunk_id in self.full_vectors:
                # Use full precision for more accurate score
                full_vector = self.full_vectors[chunk_id]
                final_similarity = self.compute_similarity(query, full_vector)
            else:
                # Fall back to approximate similarity
                final_similarity = similarity

            if self.metric == Metric.COSINE:
                distance = 1.0 - final_similarity
            elif self.metric == Metric.DOT_PRODUCT:
                distance = -final_similarity
            else:  # Euclidean
                distance = -final_similarity

            final_results.append((final_similarity, SearchResult(
                chunk_id=chunk_id,
                score=final_similarity,
                distance=distance
            )))

        # Sort by final similarity and return top k
        final_results.sort(reverse=True)
        return [result for _, result in final_results[:k]]

    def search_with_adaptive_probes(
        self,
        query_vector: List[float],
        k: int,
        min_candidates: int = None
    ) -> List[SearchResult]:
        """
        Search with adaptive probe count based on candidate availability.

        Args:
            query_vector: Query vector
            k: Number of neighbors to return
            min_candidates: Minimum candidates to collect (default: 4*k)

        Returns:
            Search results with adaptive probing
        """
        if min_candidates is None:
            min_candidates = max(k * 4, 50)

        original_nprobe = self.nprobe

        # Start with default nprobe and increase if needed
        for probe_multiplier in [1, 2, 4]:
            self.nprobe = min(original_nprobe * probe_multiplier, self.nlist)

            results = self.search(query_vector, k)

            # Estimate candidate count (rough heuristic)
            estimated_candidates = len(results) * self.nprobe // max(1, original_nprobe)

            if estimated_candidates >= min_candidates or self.nprobe >= self.nlist:
                break

        # Restore original nprobe
        self.nprobe = original_nprobe
        return results

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index."""
        removed = False

        # Search through all inverted lists
        for cluster_id in list(self.inverted_lists.keys()):
            original_len = len(self.inverted_lists[cluster_id])
            self.inverted_lists[cluster_id] = [
                (cid, code) for cid, code in self.inverted_lists[cluster_id]
                if cid != chunk_id
            ]

            if len(self.inverted_lists[cluster_id]) < original_len:
                removed = True

            # Remove empty lists
            if not self.inverted_lists[cluster_id]:
                del self.inverted_lists[cluster_id]

        # Remove from full vectors
        if chunk_id in self.full_vectors:
            del self.full_vectors[chunk_id]

        if removed:
            self.size -= 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        # Compute inverted list statistics
        list_sizes = [len(lst) for lst in self.inverted_lists.values()]
        if list_sizes:
            avg_list_size = np.mean(list_sizes)
            max_list_size = max(list_sizes)
            min_list_size = min(list_sizes)
            non_empty_lists = len(list_sizes)
        else:
            avg_list_size = max_list_size = min_list_size = non_empty_lists = 0

        # Estimate memory usage
        coarse_memory = self.nlist * self.dimension * 4 if self.coarse_centroids is not None else 0
        pq_memory = self.pq.m * self.pq.ksub * self.pq.dsub * 4 if self.pq.is_trained else 0
        codes_memory = self.size * self.pq.m  # uint8 codes
        full_vectors_memory = len(self.full_vectors) * self.dimension * 4

        compression_ratio = 1.0
        if self.size > 0:
            uncompressed_size = self.size * self.dimension * 4
            compressed_size = codes_memory + full_vectors_memory
            compression_ratio = uncompressed_size / max(compressed_size, 1)

        return {
            "type": "ivf_pq",
            "dimension": self.dimension,
            "size": self.size,
            "metric": self.metric.value,
            "normalized": self.normalize,
            "is_built": self.is_built,
            "is_trained": self.is_trained,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "pq_m": self.pq.m,
            "pq_nbits": self.pq.nbits,
            "pq_ksub": self.pq.ksub,
            "rerank_size": self.rerank_size,
            "non_empty_lists": non_empty_lists,
            "avg_list_size": round(avg_list_size, 2),
            "max_list_size": max_list_size,
            "min_list_size": min_list_size,
            "full_vectors_cached": len(self.full_vectors),
            "compression_ratio": round(compression_ratio, 2),
            "memory_usage_bytes": coarse_memory + pq_memory + codes_memory + full_vectors_memory,
            "memory_savings_vs_full": f"{(1 - 1/compression_ratio)*100:.1f}%" if compression_ratio > 1 else "0%"
        }