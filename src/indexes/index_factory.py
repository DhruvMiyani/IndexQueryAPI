"""
Factory for creating index instances.

Provides a simple interface to create different index types.
"""

from typing import Union

from .base import BaseIndex, IndexType
from .linear_index import LinearIndex
from .kd_tree_index import KDTreeIndex
from .lsh_index import LSHIndex


class IndexFactory:
    """Factory for creating vector index instances."""

    @staticmethod
    def create(
        index_type: IndexType,
        dimension: int,
        **kwargs: Union[int, str, bool, float]
    ) -> BaseIndex:
        """
        Create an index instance.

        Args:
            index_type: Type of index to create
            dimension: Vector dimension
            **kwargs: Additional parameters for specific index types

        Returns:
            Index instance

        Raises:
            ValueError: If index type is not supported
        """
        if index_type == IndexType.LINEAR:
            return LinearIndex(dimension)

        elif index_type == IndexType.KD_TREE:
            return KDTreeIndex(dimension)

        elif index_type == IndexType.LSH:
            num_hyperplanes: int = int(kwargs.get("num_hyperplanes") or 10)
            num_tables: int = int(kwargs.get("num_tables") or 5)
            return LSHIndex(dimension, num_hyperplanes, num_tables)

        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    @staticmethod
    def recommend_index_type(
        dimension: int,
        dataset_size: int,
        accuracy_required: bool = True
    ) -> IndexType:
        """
        Recommend an index type based on data characteristics.

        Args:
            dimension: Vector dimension
            dataset_size: Number of vectors
            accuracy_required: Whether exact results are needed

        Returns:
            Recommended index type
        """
        # Small dataset - linear is fine
        if dataset_size < 1000:
            return IndexType.LINEAR

        # Low dimension and exact results needed - KD-Tree
        if dimension <= 20 and accuracy_required:
            return IndexType.KD_TREE

        # High dimension or approximate OK - LSH
        if dimension > 20 or not accuracy_required:
            return IndexType.LSH

        # Default to linear for exact results
        return IndexType.LINEAR