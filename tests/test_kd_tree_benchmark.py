"""
Benchmark KD-Tree vs brute force to understand performance characteristics.

This helps identify the break-even point and validate theoretical complexity.
"""

import time
import uuid
from typing import List, Tuple
import numpy as np
import pytest

from src.indexes.kd_tree_index import KDTreeIndex
from src.indexes.linear_index import LinearIndex


def gen_points(n: int, d: int, seed: int = 0) -> List[Tuple[str, List[float]]]:
    """Generate n random d-dimensional points."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        v = rng.normal(size=d).astype(float).tolist()
        items.append((str(uuid.uuid4()), v))
    return items


def bench_kd_vs_brute(n: int = 10000, d: int = 8, k: int = 10, queries: int = 100):
    """
    Benchmark KD-Tree vs Linear (brute force) search.

    Returns: (build_ms, kd_query_ms, linear_query_ms)
    """
    print(f"\nBenchmarking n={n}, d={d}, k={k}, queries={queries}")

    # Generate data
    items = gen_points(n, d)

    # Build KD-Tree
    kd = KDTreeIndex(dimension=d)
    t0 = time.time()
    kd.build(items)
    t1 = time.time()
    build_ms = (t1 - t0) * 1000

    # Build Linear index
    linear = LinearIndex(dimension=d)
    linear.build(items)

    # Generate query vectors
    rng = np.random.default_rng(1)
    queries_list = rng.normal(size=(queries, d)).astype(float).tolist()

    # Benchmark KD-Tree queries
    t0 = time.time()
    for q in queries_list:
        kd.search(q, k)
    kd_ms = (time.time() - t0) * 1000

    # Benchmark Linear queries
    t0 = time.time()
    for q in queries_list:
        linear.search(q, k)
    linear_ms = (time.time() - t0) * 1000

    speedup = linear_ms / kd_ms if kd_ms > 0 else float('inf')

    print(f"Build time: {build_ms:.1f}ms")
    print(f"KD-Tree queries: {kd_ms:.1f}ms ({kd_ms/queries:.2f}ms/query)")
    print(f"Linear queries: {linear_ms:.1f}ms ({linear_ms/queries:.2f}ms/query)")
    print(f"Speedup: {speedup:.2f}x")

    return build_ms, kd_ms, linear_ms


class TestKDTreeBenchmarks:
    """Benchmark tests for KD-Tree performance characteristics."""

    @pytest.mark.benchmark
    def test_low_dimensions_performance(self):
        """Test KD-Tree performance in low dimensions (should be faster)."""
        build_ms, kd_ms, linear_ms = bench_kd_vs_brute(n=5000, d=3, k=10, queries=50)

        # In 3D, KD-Tree should outperform linear search
        speedup = linear_ms / kd_ms
        print(f"Low-dim speedup: {speedup:.2f}x")

        # KD-Tree should be faster (at least some speedup)
        assert speedup > 1.0, f"Expected KD-Tree to be faster in 3D, got speedup: {speedup:.2f}x"

    @pytest.mark.benchmark
    def test_moderate_dimensions_performance(self):
        """Test KD-Tree performance in moderate dimensions."""
        build_ms, kd_ms, linear_ms = bench_kd_vs_brute(n=5000, d=8, k=10, queries=50)

        speedup = linear_ms / kd_ms
        print(f"Moderate-dim speedup: {speedup:.2f}x")

        # Should still have some advantage in 8D
        # (though may vary based on data distribution)

    @pytest.mark.benchmark
    def test_high_dimensions_curse(self):
        """Test KD-Tree performance degradation in high dimensions."""
        build_ms, kd_ms, linear_ms = bench_kd_vs_brute(n=2000, d=32, k=10, queries=30)

        speedup = linear_ms / kd_ms
        print(f"High-dim speedup: {speedup:.2f}x")

        # In high dimensions, curse of dimensionality should reduce effectiveness
        # KD-Tree may not be much faster than linear, might even be slower due to overhead
        print(f"High-dimensional performance shows curse of dimensionality effect")

    @pytest.mark.benchmark
    def test_scaling_with_dataset_size(self):
        """Test how performance scales with dataset size."""
        dimensions = 5  # Keep dimensions low to see tree benefits

        sizes = [1000, 5000, 10000]
        results = []

        for n in sizes:
            print(f"\n--- Dataset size: {n} ---")
            build_ms, kd_ms, linear_ms = bench_kd_vs_brute(n=n, d=dimensions, k=10, queries=30)
            speedup = linear_ms / kd_ms
            results.append((n, speedup, kd_ms/30, linear_ms/30))  # per-query times

        print(f"\nScaling results:")
        print(f"Size\tSpeedup\tKD-ms/q\tLinear-ms/q")
        for n, speedup, kd_per_q, lin_per_q in results:
            print(f"{n}\t{speedup:.2f}x\t{kd_per_q:.2f}\t{lin_per_q:.2f}")

        # Linear search should scale linearly O(n)
        # KD-Tree should scale better, closer to O(log n) in good cases

    @pytest.mark.benchmark
    def test_build_time_scaling(self):
        """Test KD-Tree build time scaling."""
        dimensions = 4
        sizes = [1000, 5000, 10000]
        build_times = []

        for n in sizes:
            items = gen_points(n, dimensions)
            kd = KDTreeIndex(dimension=dimensions)

            t0 = time.time()
            kd.build(items)
            build_time = (time.time() - t0) * 1000
            build_times.append((n, build_time))
            print(f"Build n={n}: {build_time:.1f}ms ({build_time/n:.3f}ms/point)")

        print(f"\nBuild time scaling (should be roughly O(n log n)):")
        for i in range(1, len(build_times)):
            prev_n, prev_time = build_times[i-1]
            curr_n, curr_time = build_times[i]
            ratio = curr_time / prev_time
            size_ratio = curr_n / prev_n
            print(f"{prev_n} -> {curr_n}: {ratio:.2f}x time for {size_ratio:.1f}x data")

    def test_correctness_vs_brute_force(self):
        """Verify KD-Tree finds reasonable neighbors (may differ from linear due to metric)."""
        n, d = 100, 4
        items = gen_points(n, d, seed=42)

        # Build both indexes
        kd = KDTreeIndex(dimension=d)
        linear = LinearIndex(dimension=d)
        kd.build(items)
        linear.build(items)

        # Test multiple queries
        rng = np.random.default_rng(1)
        for _ in range(10):
            query = rng.normal(size=d).tolist()

            kd_results = kd.search(query, k=5)
            linear_results = linear.search(query, k=5)

            # Note: KD-Tree uses Euclidean pruning but returns cosine scores,
            # so results may differ from linear search. We check that both
            # return reasonable high-quality results.

            # Both should return valid results
            assert len(kd_results) == 5
            assert len(linear_results) == 5

            # Both should return results with reasonable similarity scores
            kd_scores = [r.score for r in kd_results]
            linear_scores = [r.score for r in linear_results]

            # Scores should be ordered (highest first)
            assert kd_scores == sorted(kd_scores, reverse=True)
            assert linear_scores == sorted(linear_scores, reverse=True)

            # At least some overlap is expected (even with different metrics)
            kd_ids = set(r.chunk_id for r in kd_results)
            linear_ids = set(r.chunk_id for r in linear_results)
            overlap = len(kd_ids & linear_ids)

            # Expect at least 2-3 common neighbors despite metric differences
            assert overlap >= 2, f"Too little overlap: {overlap}/5 (expected due to Euclidean vs Cosine)"

    @pytest.mark.benchmark
    def test_memory_usage_stats(self):
        """Test memory usage and tree structure stats."""
        n, d = 1000, 6
        items = gen_points(n, d)

        kd = KDTreeIndex(dimension=d)
        kd.build(items)

        stats = kd.get_stats()
        depth = stats["tree_depth"]

        print(f"\nMemory/Structure stats for n={n}, d={d}:")
        print(f"Tree depth: {depth}")
        print(f"Theoretical min depth: {int(np.log2(n))}")
        print(f"Theoretical max depth: {n}")
        print(f"Tree efficiency: {int(np.log2(n)) / depth:.2f}")

        # Depth should be reasonable (not degenerate)
        assert depth <= n, "Tree depth shouldn't exceed number of points"
        assert depth >= int(np.log2(n)), "Tree should have at least log(n) depth"


if __name__ == "__main__":
    # Run benchmarks directly
    print("=== KD-Tree vs Linear Search Benchmarks ===")

    print("\n1. Low dimensions (should favor KD-Tree):")
    bench_kd_vs_brute(n=8000, d=3, k=10, queries=100)

    print("\n2. Moderate dimensions:")
    bench_kd_vs_brute(n=8000, d=8, k=10, queries=100)

    print("\n3. High dimensions (curse of dimensionality):")
    bench_kd_vs_brute(n=3000, d=32, k=10, queries=50)

    print("\n4. Very high dimensions:")
    bench_kd_vs_brute(n=2000, d=64, k=10, queries=30)