"""
Advanced factory for creating optimized index instances.

Provides intelligent index selection based on dataset characteristics
and performance requirements. Includes all optimized index implementations.
"""

from typing import Dict, Any, Optional
import math

from .base import BaseIndex, IndexType, Metric
from .linear_index import LinearIndex
from .optimized_linear_index import OptimizedLinearIndex
from .kd_tree_index import KDTreeIndex
from .improved_kd_tree_index import ImprovedKDTreeIndex
from .lsh_index import LSHIndex
from .multiprobe_lsh_index import MultiProbeLSHIndex
from .hnsw_index import HNSWIndex
from .ivf_pq_index import IVFPQIndex


class AdvancedIndexFactory:
    """
    Advanced factory for creating optimized vector index instances.

    Provides intelligent recommendations based on:
    - Dataset size and dimensionality
    - Accuracy vs speed requirements
    - Memory constraints
    - Query pattern expectations
    """

    # Performance thresholds for recommendations
    SMALL_DATASET = 1_000
    MEDIUM_DATASET = 100_000
    LARGE_DATASET = 1_000_000
    VERY_LARGE_DATASET = 10_000_000

    LOW_DIMENSION = 16
    MEDIUM_DIMENSION = 128
    HIGH_DIMENSION = 512

    @staticmethod
    def create(
        index_type: IndexType,
        dimension: int,
        metric: Metric = Metric.COSINE,
        normalize: bool = True,
        **kwargs
    ) -> BaseIndex:
        """
        Create an optimized index instance.

        Args:
            index_type: Type of index to create
            dimension: Vector dimension
            metric: Distance/similarity metric
            normalize: Whether to normalize vectors
            **kwargs: Additional parameters for specific index types

        Returns:
            Optimized index instance

        Raises:
            ValueError: If index type is not supported
        """
        if index_type == IndexType.LINEAR:
            # Use optimized linear index by default
            use_optimized = kwargs.get("use_optimized", True)
            if use_optimized:
                return OptimizedLinearIndex(dimension, metric, normalize)
            else:
                return LinearIndex(dimension)

        elif index_type == IndexType.KD_TREE:
            # Use improved KD-Tree by default
            use_improved = kwargs.get("use_improved", True)
            if use_improved:
                warn_high_dim = kwargs.get("warn_high_dimension", True)
                return ImprovedKDTreeIndex(dimension, metric, normalize, warn_high_dim)
            else:
                return KDTreeIndex(dimension)

        elif index_type == IndexType.LSH:
            # Use multi-probe LSH by default
            use_multiprobe = kwargs.get("use_multiprobe", True)
            if use_multiprobe:
                return MultiProbeLSHIndex(
                    dimension=dimension,
                    num_tables=kwargs.get("num_tables", 8),
                    num_hyperplanes=kwargs.get("num_hyperplanes", 32),
                    max_probes=kwargs.get("max_probes", 8),
                    candidate_limit=kwargs.get("candidate_limit", 1000),
                    seed=kwargs.get("seed", 42),
                    metric=metric,
                    normalize=normalize
                )
            else:
                return LSHIndex(
                    dimension=dimension,
                    num_tables=kwargs.get("num_tables", 10),
                    num_hyperplanes=kwargs.get("num_hyperplanes", 16)
                )

        elif index_type == IndexType.HNSW:
            return HNSWIndex(
                dimension=dimension,
                M=kwargs.get("M", 16),
                ef_construction=kwargs.get("ef_construction", 200),
                ef_search=kwargs.get("ef_search", 50),
                max_M=kwargs.get("max_M"),
                max_M0=kwargs.get("max_M0"),
                seed=kwargs.get("seed"),
                metric=metric,
                normalize=normalize
            )

        elif index_type == IndexType.IVF_PQ:
            return IVFPQIndex(
                dimension=dimension,
                nlist=kwargs.get("nlist", 4096),
                m=kwargs.get("m", 8),
                nbits=kwargs.get("nbits", 8),
                nprobe=kwargs.get("nprobe", 8),
                rerank_size=kwargs.get("rerank_size", 64),
                metric=metric,
                normalize=normalize,
                seed=kwargs.get("seed", 42)
            )

        # Explicit optimized versions
        elif index_type == IndexType.OPTIMIZED_LINEAR:
            return OptimizedLinearIndex(dimension, metric, normalize)

        elif index_type == IndexType.IMPROVED_KD_TREE:
            warn_high_dim = kwargs.get("warn_high_dimension", True)
            return ImprovedKDTreeIndex(dimension, metric, normalize, warn_high_dim)

        elif index_type == IndexType.MULTIPROBE_LSH:
            return MultiProbeLSHIndex(
                dimension=dimension,
                num_tables=kwargs.get("num_tables", 8),
                num_hyperplanes=kwargs.get("num_hyperplanes", 32),
                max_probes=kwargs.get("max_probes", 8),
                candidate_limit=kwargs.get("candidate_limit", 1000),
                seed=kwargs.get("seed", 42),
                metric=metric,
                normalize=normalize
            )

        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    @staticmethod
    def recommend_index_type(
        dimension: int,
        dataset_size: int,
        accuracy_required: bool = True,
        memory_constrained: bool = False,
        high_throughput: bool = False,
        dynamic_updates: bool = False,
        query_pattern: str = "mixed"  # "point", "batch", "mixed"
    ) -> IndexType:
        """
        Recommend optimal index type based on comprehensive characteristics.

        Args:
            dimension: Vector dimension
            dataset_size: Number of vectors in dataset
            accuracy_required: Whether high accuracy is critical
            memory_constrained: Whether memory usage is a major concern
            high_throughput: Whether high query throughput is needed
            dynamic_updates: Whether frequent insertions/deletions are expected
            query_pattern: Expected query pattern ("point", "batch", "mixed")

        Returns:
            Recommended index type with reasoning
        """
        factory = AdvancedIndexFactory

        # Very small datasets - always use linear
        if dataset_size < factory.SMALL_DATASET:
            return IndexType.LINEAR

        # Memory-constrained large datasets
        if memory_constrained and dataset_size > factory.LARGE_DATASET:
            return IndexType.IVF_PQ

        # Low dimension with high accuracy requirements
        if (dimension <= factory.LOW_DIMENSION and
            accuracy_required and
            dataset_size <= factory.MEDIUM_DATASET):
            return IndexType.KD_TREE

        # High throughput batch queries with large datasets
        if (high_throughput and
            query_pattern == "batch" and
            dataset_size > factory.MEDIUM_DATASET):
            return IndexType.LINEAR  # Optimized linear excels at batch operations

        # Dynamic updates with good performance
        if dynamic_updates and dataset_size > factory.SMALL_DATASET:
            return IndexType.HNSW

        # Very large datasets
        if dataset_size > factory.VERY_LARGE_DATASET:
            if memory_constrained:
                return IndexType.IVF_PQ
            else:
                return IndexType.HNSW

        # Large datasets with good accuracy/speed balance
        if dataset_size > factory.LARGE_DATASET:
            return IndexType.HNSW

        # Medium datasets, high dimensions
        if dataset_size > factory.MEDIUM_DATASET and dimension > factory.MEDIUM_DIMENSION:
            if accuracy_required:
                return IndexType.HNSW
            else:
                return IndexType.LSH

        # Medium datasets, moderate dimensions
        if dataset_size > factory.MEDIUM_DATASET:
            return IndexType.HNSW

        # Small-medium datasets, approximate search OK
        if not accuracy_required:
            return IndexType.LSH

        # Default: optimized linear for exact search
        return IndexType.LINEAR

    @staticmethod
    def get_recommended_parameters(
        index_type: IndexType,
        dimension: int,
        dataset_size: int,
        **constraints
    ) -> Dict[str, Any]:
        """
        Get recommended parameters for a specific index type.

        Args:
            index_type: The index type
            dimension: Vector dimension
            dataset_size: Dataset size
            **constraints: Additional constraints (memory_limit, accuracy_target, etc.)

        Returns:
            Dictionary of recommended parameters
        """
        factory = AdvancedIndexFactory

        if index_type == IndexType.LINEAR:
            return {
                "use_optimized": True,
                "description": "Optimized linear search with vectorization"
            }

        elif index_type == IndexType.KD_TREE:
            return {
                "use_improved": True,
                "warn_high_dimension": dimension > 20,
                "description": f"Improved KD-Tree (optimal for dim <= 20, current: {dimension})"
            }

        elif index_type == IndexType.LSH:
            # Scale parameters with dataset size and dimension
            if dataset_size < factory.MEDIUM_DATASET:
                num_tables, num_hyperplanes, max_probes = 4, 16, 4
            elif dataset_size < factory.LARGE_DATASET:
                num_tables, num_hyperplanes, max_probes = 8, 32, 8
            else:
                num_tables, num_hyperplanes, max_probes = 12, 48, 12

            # Adjust for dimension
            if dimension > factory.HIGH_DIMENSION:
                num_hyperplanes = min(64, num_hyperplanes + 16)

            return {
                "use_multiprobe": True,
                "num_tables": num_tables,
                "num_hyperplanes": num_hyperplanes,
                "max_probes": max_probes,
                "candidate_limit": min(2000, dataset_size // 10),
                "description": f"Multi-probe LSH optimized for {dataset_size} vectors"
            }

        elif index_type == IndexType.HNSW:
            # Scale M with dimension and dataset size
            if dimension <= factory.LOW_DIMENSION:
                M = 8
            elif dimension <= factory.MEDIUM_DIMENSION:
                M = 16
            else:
                M = 32

            # Scale ef_construction with dataset size
            if dataset_size < factory.MEDIUM_DATASET:
                ef_construction = 100
            elif dataset_size < factory.LARGE_DATASET:
                ef_construction = 200
            else:
                ef_construction = 400

            # Default ef_search (can be tuned at query time)
            ef_search = max(50, M * 2)

            return {
                "M": M,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "description": f"HNSW optimized for {dataset_size} vectors, dim={dimension}"
            }

        elif index_type == IndexType.IVF_PQ:
            # Scale nlist with dataset size
            nlist = min(65536, max(256, int(math.sqrt(dataset_size))))

            # Choose PQ parameters
            if dimension <= 64:
                m = 8
            elif dimension <= 256:
                m = 16
            else:
                m = 32

            # Ensure m divides dimension
            while dimension % m != 0 and m > 1:
                m -= 1

            # Scale nprobe with nlist
            nprobe = max(1, min(nlist // 8, 32))

            return {
                "nlist": nlist,
                "m": m,
                "nbits": 8,
                "nprobe": nprobe,
                "rerank_size": 64,
                "description": f"IVF-PQ with {nlist} clusters, {m} subquantizers"
            }

        else:
            return {"description": "Unknown index type"}

    @staticmethod
    def create_recommended(
        dimension: int,
        dataset_size: int,
        metric: Metric = Metric.COSINE,
        normalize: bool = True,
        **constraints
    ) -> tuple[BaseIndex, str]:
        """
        Create a recommended index with optimal parameters.

        Args:
            dimension: Vector dimension
            dataset_size: Expected dataset size
            metric: Distance metric
            normalize: Whether to normalize vectors
            **constraints: Additional constraints

        Returns:
            Tuple of (index_instance, reasoning_description)
        """
        # Get recommendation
        index_type = AdvancedIndexFactory.recommend_index_type(
            dimension, dataset_size, **constraints
        )

        # Get recommended parameters
        params = AdvancedIndexFactory.get_recommended_parameters(
            index_type, dimension, dataset_size, **constraints
        )

        description = params.pop("description", "Recommended index")

        # Create index
        index = AdvancedIndexFactory.create(
            index_type, dimension, metric, normalize, **params
        )

        reasoning = (
            f"Recommended {index_type.value} for {dataset_size:,} vectors "
            f"of dimension {dimension}. {description}"
        )

        return index, reasoning

    @staticmethod
    def compare_index_types(
        dimension: int,
        dataset_size: int,
        **constraints
    ) -> Dict[IndexType, Dict[str, Any]]:
        """
        Compare all index types for given characteristics.

        Args:
            dimension: Vector dimension
            dataset_size: Dataset size
            **constraints: Additional constraints

        Returns:
            Dictionary mapping index types to their characteristics
        """
        factory = AdvancedIndexFactory
        comparison = {}

        for index_type in IndexType:
            try:
                params = factory.get_recommended_parameters(
                    index_type, dimension, dataset_size, **constraints
                )

                # Estimate characteristics
                if index_type == IndexType.LINEAR:
                    build_time = "O(1)"
                    query_time = "O(n)"
                    memory = "100%"
                    accuracy = "100%"
                    supports_updates = "Excellent"

                elif index_type == IndexType.KD_TREE:
                    build_time = "O(n log n)"
                    query_time = "O(log n)" if dimension <= 20 else "O(n) degraded"
                    memory = "100%"
                    accuracy = "100%"
                    supports_updates = "Poor"

                elif index_type == IndexType.LSH:
                    build_time = "O(n)"
                    query_time = "O(n^ρ) sub-linear"
                    memory = "150-200%"
                    accuracy = "80-95%"
                    supports_updates = "Good"

                elif index_type == IndexType.HNSW:
                    build_time = "O(n log n)"
                    query_time = "O(log n)"
                    memory = "120-150%"
                    accuracy = "95-99%"
                    supports_updates = "Excellent"

                elif index_type == IndexType.IVF_PQ:
                    build_time = "O(n)"
                    query_time = "O(√n)"
                    memory = "10-30%"
                    accuracy = "85-95%"
                    supports_updates = "Good"

                comparison[index_type] = {
                    "recommended_params": params,
                    "build_complexity": build_time,
                    "query_complexity": query_time,
                    "memory_usage": memory,
                    "typical_accuracy": accuracy,
                    "update_support": supports_updates,
                    "best_for": factory._get_best_use_case(index_type, dimension, dataset_size)
                }

            except Exception as e:
                comparison[index_type] = {"error": str(e)}

        return comparison

    @staticmethod
    def _get_best_use_case(index_type: IndexType, dimension: int, dataset_size: int) -> str:
        """Get the best use case description for an index type."""
        factory = AdvancedIndexFactory

        if index_type == IndexType.LINEAR:
            return "Small datasets, exact results, batch queries"

        elif index_type == IndexType.KD_TREE:
            if dimension <= 20:
                return "Low dimensions, exact results, static data"
            else:
                return "Not recommended for high dimensions"

        elif index_type == IndexType.LSH:
            return "High dimensions, approximate results, fast insertion"

        elif index_type == IndexType.HNSW:
            return "General purpose, good accuracy/speed balance, dynamic updates"

        elif index_type == IndexType.IVF_PQ:
            return "Large datasets, memory constrained, acceptable accuracy loss"

        return "Unknown"