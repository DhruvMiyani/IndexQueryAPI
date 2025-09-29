"""
Comprehensive tests for all optimized index implementations.

Tests functionality, performance, and correctness of all new index types.
"""

import pytest
import numpy as np
from uuid import uuid4, UUID
from typing import List, Tuple
import time

# Import all the optimized indexes
from src.indexes.base import Metric, IndexType
from src.indexes.optimized_linear_index import OptimizedLinearIndex
from src.indexes.improved_kd_tree_index import ImprovedKDTreeIndex
from src.indexes.multiprobe_lsh_index import MultiProbeLSHIndex
from src.indexes.hnsw_index import HNSWIndex
from src.indexes.ivf_pq_index import IVFPQIndex
from src.indexes.advanced_index_factory import AdvancedIndexFactory


class TestOptimizedIndexes:
    """Test suite for all optimized index implementations."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        dimension = 128
        n_vectors = 1000

        # Generate random vectors
        vectors = []
        for i in range(n_vectors):
            vec = np.random.randn(dimension).astype(np.float32)
            chunk_id = uuid4()
            vectors.append((chunk_id, vec.tolist()))

        return vectors, dimension

    @pytest.fixture
    def small_data(self):
        """Generate small test dataset."""
        np.random.seed(123)
        dimension = 32
        n_vectors = 100

        vectors = []
        for i in range(n_vectors):
            vec = np.random.randn(dimension).astype(np.float32)
            chunk_id = uuid4()
            vectors.append((chunk_id, vec.tolist()))

        return vectors, dimension

    def test_optimized_linear_index(self, sample_data):
        """Test OptimizedLinearIndex functionality."""
        vectors, dimension = sample_data

        # Test different metrics
        for metric in [Metric.COSINE, Metric.EUCLIDEAN, Metric.DOT_PRODUCT]:
            index = OptimizedLinearIndex(dimension, metric=metric)

            # Build index
            index.build(vectors)
            assert index.is_built
            assert index.size == len(vectors)

            # Test search
            query = np.random.randn(dimension).tolist()
            results = index.search(query, k=10)

            assert len(results) <= 10
            assert len(results) > 0

            # Verify results are sorted by score
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

            # Test add operation
            new_id = uuid4()
            new_vector = np.random.randn(dimension).tolist()
            index.add(new_id, new_vector)
            assert index.size == len(vectors) + 1

            # Test remove operation
            removed = index.remove(new_id)
            assert removed
            assert index.size == len(vectors)

            print(f"✅ OptimizedLinearIndex with {metric.value} passed")

    def test_optimized_linear_batch_search(self, sample_data):
        """Test batch search functionality."""
        vectors, dimension = sample_data

        index = OptimizedLinearIndex(dimension, metric=Metric.COSINE)
        index.build(vectors)

        # Test batch search
        queries = [np.random.randn(dimension).tolist() for _ in range(5)]
        batch_results = index.batch_search(queries, k=5)

        assert len(batch_results) == 5
        for results in batch_results:
            assert len(results) <= 5

        # Compare with individual searches
        individual_results = [index.search(q, k=5) for q in queries]

        # Results should be identical (within floating point precision)
        for i in range(5):
            batch_ids = [r.chunk_id for r in batch_results[i]]
            individual_ids = [r.chunk_id for r in individual_results[i]]
            assert batch_ids == individual_ids

        print("✅ OptimizedLinearIndex batch search passed")

    def test_improved_kd_tree(self, small_data):
        """Test ImprovedKDTreeIndex functionality."""
        vectors, dimension = small_data

        # Test with different metrics
        for metric in [Metric.COSINE, Metric.EUCLIDEAN]:
            index = ImprovedKDTreeIndex(dimension, metric=metric, warn_high_dimension=False)

            # Build index
            index.build(vectors)
            assert index.is_built
            assert index.size == len(vectors)

            # Test search
            query = np.random.randn(dimension).tolist()
            results = index.search(query, k=10)

            assert len(results) <= 10
            assert len(results) > 0

            # Test range search
            radius_results = index.search_radius(query, radius=1.0)
            assert isinstance(radius_results, list)

            # All results should be within radius
            for result in radius_results:
                assert result.distance <= 1.0

            print(f"✅ ImprovedKDTreeIndex with {metric.value} passed")

    def test_multiprobe_lsh(self, sample_data):
        """Test MultiProbeLSHIndex functionality."""
        vectors, dimension = sample_data

        index = MultiProbeLSHIndex(
            dimension=dimension,
            num_tables=4,
            num_hyperplanes=16,
            max_probes=4,
            metric=Metric.COSINE
        )

        # Build index
        index.build(vectors)
        assert index.is_built
        assert index.size == len(vectors)

        # Test search
        query = np.random.randn(dimension).tolist()
        results = index.search(query, k=10)

        assert len(results) <= 10
        # LSH might return fewer results due to bucketing

        # Test with different probe counts
        original_probes = index.max_probes

        # More probes should generally give more/better results
        index.max_probes = 1
        results_few_probes = index.search(query, k=10)

        index.max_probes = 8
        results_many_probes = index.search(query, k=10)

        # Restore original
        index.max_probes = original_probes

        # Test recall tuning
        recall_results = index.search_with_recall_tuning(query, k=10, target_recall=0.9)
        assert isinstance(recall_results, list)

        print("✅ MultiProbeLSHIndex passed")

    def test_hnsw_index(self, sample_data):
        """Test HNSWIndex functionality."""
        vectors, dimension = sample_data

        index = HNSWIndex(
            dimension=dimension,
            M=8,  # Smaller for faster testing
            ef_construction=50,
            ef_search=25,
            metric=Metric.COSINE
        )

        # Build index
        print("Building HNSW index...")
        start_time = time.time()
        index.build(vectors)
        build_time = time.time() - start_time

        assert index.is_built
        assert index.size == len(vectors)
        print(f"HNSW build time: {build_time:.2f}s")

        # Test search
        query = np.random.randn(dimension).tolist()
        results = index.search(query, k=10)

        assert len(results) <= 10
        assert len(results) > 0

        # Test ef_search tuning
        index.set_ef_search(50)
        results_high_ef = index.search(query, k=10)
        assert len(results_high_ef) <= 10

        # Test statistics
        stats = index.get_stats()
        assert stats["type"] == "hnsw"
        assert stats["size"] == len(vectors)
        assert "max_level" in stats
        assert "memory_usage_mb" in stats

        print("✅ HNSWIndex passed")

    def test_ivf_pq_index(self, sample_data):
        """Test IVFPQIndex functionality."""
        vectors, dimension = sample_data

        # Use smaller parameters for testing
        index = IVFPQIndex(
            dimension=dimension,
            nlist=16,  # Small for testing
            m=8,       # 8 subvectors (dimension must be divisible)
            nbits=8,
            nprobe=4,
            rerank_size=32
        )

        # Build index (includes training)
        print("Building IVF-PQ index...")
        start_time = time.time()
        index.build(vectors)
        build_time = time.time() - start_time

        assert index.is_built
        assert index.is_trained
        assert index.size == len(vectors)
        print(f"IVF-PQ build time: {build_time:.2f}s")

        # Test search
        query = np.random.randn(dimension).tolist()
        results = index.search(query, k=10)

        assert len(results) <= 10
        # IVF-PQ might return fewer results due to clustering

        # Test adaptive search
        adaptive_results = index.search_with_adaptive_probes(query, k=10)
        assert isinstance(adaptive_results, list)

        # Test statistics
        stats = index.get_stats()
        assert stats["type"] == "ivf_pq"
        assert stats["size"] == len(vectors)
        assert stats["compression_ratio"] > 1.0
        assert "memory_savings_vs_full" in stats

        print("✅ IVFPQIndex passed")

    def test_advanced_factory_recommendations(self, sample_data):
        """Test AdvancedIndexFactory recommendations."""
        vectors, dimension = sample_data

        # Test different scenarios
        scenarios = [
            {"dataset_size": 100, "accuracy_required": True},      # Should recommend Linear
            {"dataset_size": 10000, "accuracy_required": True},   # Should recommend HNSW
            {"dataset_size": 1000000, "memory_constrained": True}, # Should recommend IVF_PQ
            {"dataset_size": 50000, "accuracy_required": False},  # Should recommend LSH
        ]

        for scenario in scenarios:
            # Get recommendation
            index_type = AdvancedIndexFactory.recommend_index_type(
                dimension=dimension, **scenario
            )

            # Get parameters
            params = AdvancedIndexFactory.get_recommended_parameters(
                index_type, dimension, scenario["dataset_size"]
            )

            print(f"Scenario {scenario} → {index_type.value}")
            print(f"  Parameters: {params}")

            # Test creating the recommended index
            index = AdvancedIndexFactory.create(
                index_type, dimension, **{k: v for k, v in params.items() if k != "description"}
            )

            assert index is not None
            assert index.dimension == dimension

        print("✅ AdvancedIndexFactory recommendations passed")

    def test_index_comparison(self, small_data):
        """Test index comparison functionality."""
        vectors, dimension = small_data

        comparison = AdvancedIndexFactory.compare_index_types(
            dimension=dimension,
            dataset_size=len(vectors)
        )

        # Should have entries for all index types
        assert len(comparison) == len(IndexType)

        for index_type, info in comparison.items():
            print(f"\n{index_type.value}:")
            if "error" in info:
                print(f"  Error: {info['error']}")
            else:
                print(f"  Build: {info['build_complexity']}")
                print(f"  Query: {info['query_complexity']}")
                print(f"  Memory: {info['memory_usage']}")
                print(f"  Accuracy: {info['typical_accuracy']}")
                print(f"  Best for: {info['best_for']}")

        print("✅ Index comparison passed")

    def test_metric_consistency(self, small_data):
        """Test that all indexes handle metrics consistently."""
        vectors, dimension = small_data

        # Take a subset for faster testing
        test_vectors = vectors[:50]
        query = np.random.randn(dimension).tolist()

        # Test linear index as ground truth
        linear_index = OptimizedLinearIndex(dimension, metric=Metric.COSINE, normalize=True)
        linear_index.build(test_vectors)
        linear_results = linear_index.search(query, k=5)
        linear_scores = [r.score for r in linear_results]

        # Test KD-Tree with same metric
        kd_index = ImprovedKDTreeIndex(dimension, metric=Metric.COSINE, normalize=True, warn_high_dimension=False)
        kd_index.build(test_vectors)
        kd_results = kd_index.search(query, k=5)
        kd_scores = [r.score for r in kd_results]

        # Scores should be very similar (allowing for small floating point differences)
        for linear_score, kd_score in zip(linear_scores, kd_scores):
            assert abs(linear_score - kd_score) < 1e-5, f"Score mismatch: {linear_score} vs {kd_score}"

        print("✅ Metric consistency test passed")

    def test_create_recommended_index(self, sample_data):
        """Test creating recommended index with full workflow."""
        vectors, dimension = sample_data

        # Create recommended index
        index, reasoning = AdvancedIndexFactory.create_recommended(
            dimension=dimension,
            dataset_size=len(vectors),
            accuracy_required=True,
            memory_constrained=False
        )

        print(f"Recommended: {reasoning}")

        # Test the recommended index
        index.build(vectors[:100])  # Use subset for faster testing
        assert index.is_built

        query = np.random.randn(dimension).tolist()
        results = index.search(query, k=5)

        assert len(results) <= 5
        assert all(isinstance(r.chunk_id, UUID) for r in results)

        # Test statistics
        stats = index.get_stats()
        assert "type" in stats
        assert "size" in stats

        print("✅ Recommended index creation passed")


if __name__ == "__main__":
    # Run specific tests
    test_instance = TestOptimizedIndexes()

    # Generate test data
    sample_data = test_instance.sample_data()
    small_data = test_instance.small_data()

    print("🚀 Starting comprehensive index tests...\n")

    try:
        # Run all tests
        test_instance.test_optimized_linear_index(sample_data)
        test_instance.test_optimized_linear_batch_search(sample_data)
        test_instance.test_improved_kd_tree(small_data)
        test_instance.test_multiprobe_lsh(sample_data)
        test_instance.test_hnsw_index(sample_data)
        test_instance.test_ivf_pq_index(sample_data)
        test_instance.test_advanced_factory_recommendations(sample_data)
        test_instance.test_index_comparison(small_data)
        test_instance.test_metric_consistency(small_data)
        test_instance.test_create_recommended_index(sample_data)

        print("\n🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()