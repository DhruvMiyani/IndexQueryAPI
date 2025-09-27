"""
Comprehensive tests for KD-Tree Index implementation.

Tests correctness, edge cases, and performance characteristics.
"""

import numpy as np
import pytest
import uuid
from uuid import UUID
from typing import List, Tuple

from src.indexes.kd_tree_index import KDTreeIndex
from src.indexes.base import SearchResult


def mk(idi: int, v: List[float]) -> Tuple[UUID, List[float]]:
    """Helper to create (uuid, vector) pairs."""
    return (uuid.uuid4(), list(map(float, v)))


def brute_force_euclidean(q: List[float], items: List[Tuple[UUID, List[float]]], k: int = 3) -> List[UUID]:
    """Brute force k-NN using Euclidean distance for comparison."""
    dists = []
    q = np.asarray(q, float)
    for cid, vec in items:
        v = np.asarray(vec, float)
        d = np.sum((q - v)**2)
        dists.append((d, cid))
    dists.sort(key=lambda x: x[0])
    return [cid for _, cid in dists[:k]]


class TestKDTreeIndex:
    """Test suite for KD-Tree Index."""

    def test_empty_tree(self):
        """Test operations on empty tree."""
        kd = KDTreeIndex(dimension=3)
        kd.build([])

        assert kd.size == 0
        assert kd.is_built
        assert kd.search([0, 0, 0], 5) == []
        assert kd.get_stats()["tree_depth"] == 0

    def test_dimension_mismatch_build(self):
        """Test dimension validation during build."""
        kd = KDTreeIndex(dimension=3)
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            kd.build([(uuid.uuid4(), [1.0, 2.0])])  # wrong length

    def test_dimension_mismatch_search(self):
        """Test dimension validation during search."""
        kd = KDTreeIndex(dimension=2)
        kd.build([mk(0, [1.0, 2.0])])

        with pytest.raises(ValueError, match="Query vector dimension mismatch"):
            kd.search([1.0, 2.0, 3.0], 1)  # wrong query dimension

    def test_dimension_mismatch_add(self):
        """Test dimension validation during add."""
        kd = KDTreeIndex(dimension=2)
        kd.build([])

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            kd.add(uuid.uuid4(), [1.0, 2.0, 3.0])  # wrong dimension

    def test_small_exact_agrees_with_bruteforce(self):
        """Test that KD-Tree matches brute force on small low-dim data."""
        # Create simple 2D points in a line
        pts = [mk(i, [i, 0]) for i in range(10)]
        kd = KDTreeIndex(dimension=2)
        kd.build(pts)

        q = [2.1, 0.0]
        top_kd = [r.chunk_id for r in kd.search(q, k=3)]
        top_bf = brute_force_euclidean(q, pts, k=3)

        # Should find same neighbors (though order might differ for ties)
        assert set(top_kd) == set(top_bf)

    def test_single_point(self):
        """Test with a single point."""
        kd = KDTreeIndex(dimension=2)
        point_id = uuid.uuid4()
        kd.build([(point_id, [1.0, 2.0])])

        results = kd.search([1.1, 2.1], k=1)
        assert len(results) == 1
        assert results[0].chunk_id == point_id
        assert results[0].score > 0.9  # Should be high similarity

    def test_add_then_search(self):
        """Test adding points dynamically then searching."""
        kd = KDTreeIndex(dimension=2)
        kd.build([])

        ids = []
        for x in [0, 1, 2, 3]:
            cid = uuid.uuid4()
            ids.append(cid)
            kd.add(cid, [float(x), 0.0])

        assert kd.size == 4

        # Search near point [2, 0]
        res = kd.search([2.1, 0.0], 2)
        assert len(res) == 2

        # Should find the two closest points
        result_ids = [r.chunk_id for r in res]
        assert ids[2] in result_ids  # point [2, 0] should be closest

    def test_remove_operations(self):
        """Test removal of points from tree."""
        kd = KDTreeIndex(dimension=2)

        # Create some points
        points = [mk(i, [i, 0]) for i in range(5)]
        kd.build(points)
        assert kd.size == 5

        # Remove middle point
        target_id = points[2][0]
        removed = kd.remove(target_id)
        assert removed is True
        assert kd.size == 4

        # Try to remove same point again
        removed_again = kd.remove(target_id)
        assert removed_again is False
        assert kd.size == 4

        # Search should still work
        results = kd.search([2.0, 0.0], k=2)
        assert len(results) == 2
        # The removed point should not appear
        result_ids = [r.chunk_id for r in results]
        assert target_id not in result_ids

    def test_depth_and_stats(self):
        """Test tree depth calculation and stats."""
        kd = KDTreeIndex(dimension=2)

        # Build with a few points
        points = [mk(i, [i, 0]) for i in range(7)]  # 7 points
        kd.build(points)

        stats = kd.get_stats()
        assert stats["type"] == "kd_tree"
        assert stats["dimension"] == 2
        assert stats["size"] == 7
        assert stats["is_built"] is True
        assert stats["tree_depth"] >= 1
        assert stats["tree_depth"] <= 7  # Can't be deeper than number of points
        assert stats["recommended_max_dim"] == 20

    def test_balanced_tree_depth(self):
        """Test that tree depth is reasonable for balanced data."""
        kd = KDTreeIndex(dimension=2)

        # Create 15 points (should give depth around log2(15) ≈ 4)
        points = [mk(i, [i % 4, i // 4]) for i in range(15)]
        kd.build(points)

        depth = kd.get_stats()["tree_depth"]
        # For 15 points, expect depth between 4 and 8 (reasonable range)
        assert 3 <= depth <= 8

    def test_higher_dimensions(self):
        """Test KD-Tree in higher dimensions (where it should still work but less efficiently)."""
        kd = KDTreeIndex(dimension=5)

        # Create some 5D points
        np.random.seed(42)
        points = []
        for i in range(20):
            vec = np.random.randn(5).tolist()
            points.append((uuid.uuid4(), vec))

        kd.build(points)
        assert kd.size == 20

        # Search should still work
        query = np.random.randn(5).tolist()
        results = kd.search(query, k=5)
        assert len(results) == 5

        # Results should be sorted by similarity (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_points(self):
        """Test handling of duplicate points."""
        kd = KDTreeIndex(dimension=2)

        # Create some duplicate points
        id1, id2 = uuid.uuid4(), uuid.uuid4()
        points = [
            (id1, [1.0, 1.0]),
            (id2, [1.0, 1.0]),  # Same coordinates
            (uuid.uuid4(), [2.0, 2.0])
        ]

        kd.build(points)
        results = kd.search([1.0, 1.0], k=2)

        # Should find both duplicate points with high similarity
        assert len(results) == 2
        assert all(r.score > 0.99 for r in results)  # Both should be very similar

    def test_zero_k(self):
        """Test search with k=0."""
        kd = KDTreeIndex(dimension=2)
        kd.build([mk(0, [1.0, 1.0])])

        results = kd.search([1.0, 1.0], k=0)
        assert len(results) == 0

    def test_k_larger_than_dataset(self):
        """Test search with k larger than available points."""
        kd = KDTreeIndex(dimension=2)
        points = [mk(i, [i, 0]) for i in range(3)]
        kd.build(points)

        results = kd.search([1.0, 0.0], k=10)  # Ask for more than available
        assert len(results) == 3  # Should return all available

    def test_consistency_after_modifications(self):
        """Test that tree remains consistent after adds and removes."""
        kd = KDTreeIndex(dimension=2)

        # Start with some points
        initial_points = [mk(i, [i, 0]) for i in range(5)]
        kd.build(initial_points)

        # Add some more points
        for i in range(5, 8):
            kd.add(uuid.uuid4(), [float(i), 0.0])

        # Remove some points
        kd.remove(initial_points[1][0])  # Remove second point

        # Tree should still work correctly
        results = kd.search([3.0, 0.0], k=3)
        assert len(results) == 3

        # Verify results are reasonable
        assert all(r.score > 0 for r in results)
        assert results[0].score >= results[1].score >= results[2].score

    def test_large_tree_structure(self):
        """Test with a larger tree to verify structure integrity."""
        kd = KDTreeIndex(dimension=3)

        # Create 100 random 3D points
        np.random.seed(123)
        points = []
        for i in range(100):
            vec = np.random.randn(3).tolist()
            points.append((uuid.uuid4(), vec))

        kd.build(points)
        assert kd.size == 100

        # Test multiple queries
        for _ in range(10):
            query = np.random.randn(3).tolist()
            results = kd.search(query, k=10)
            assert len(results) == 10

            # Verify ordering
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_edge_case_all_same_points(self):
        """Test with all points at same location."""
        kd = KDTreeIndex(dimension=2)

        # All points at origin (but avoid true zero vectors due to cosine issues)
        points = [(uuid.uuid4(), [1.0, 1.0]) for _ in range(5)]
        kd.build(points)

        results = kd.search([1.0, 1.0], k=3)
        assert len(results) == 3
        # All should have perfect similarity (cosine of identical vectors = 1)
        assert all(abs(r.score - 1.0) < 1e-6 for r in results)