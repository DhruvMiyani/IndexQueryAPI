"""
Persistence module for vector database.

Provides durability and recovery capabilities.
"""

from .persistence_manager import PersistenceManager

__all__ = ["PersistenceManager"]