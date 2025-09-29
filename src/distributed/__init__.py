"""
Distributed architecture module for vector database.

Provides high availability and scalability through leader-follower replication.
"""

from .distributed_architecture import (
    DistributedCoordinator,
    NodeRole,
    NodeStatus,
    NodeInfo,
    ReplicationMode,
    ReplicationLog,
)

__all__ = [
    "DistributedCoordinator",
    "NodeRole",
    "NodeStatus",
    "NodeInfo",
    "ReplicationMode",
    "ReplicationLog",
]