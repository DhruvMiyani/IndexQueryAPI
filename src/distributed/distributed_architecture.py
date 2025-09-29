"""
Leader-Follower distributed architecture for vector database.

Provides high availability and read scalability through replication.
Design choices:
- Leader handles all writes for consistency
- Followers handle read queries for scalability
- Automatic leader election via consensus (simplified using etcd/Redis)
- Asynchronous replication with eventual consistency
- Health checks and automatic failover
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeRole(str, Enum):
    """Node roles in distributed system."""

    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"  # During election


class NodeStatus(str, Enum):
    """Node health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ReplicationMode(str, Enum):
    """Replication modes."""

    ASYNC = "async"  # Default: better performance
    SYNC = "sync"  # Strong consistency but slower


class NodeInfo(BaseModel):
    """Information about a cluster node."""

    node_id: str = Field(description="Unique node identifier")
    address: str = Field(description="Node network address")
    port: int = Field(description="Node port")
    role: NodeRole = Field(description="Current node role")
    status: NodeStatus = Field(description="Node health status")
    last_heartbeat: datetime = Field(description="Last heartbeat time")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReplicationLog(BaseModel):
    """Replication log entry."""

    log_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation: str = Field(description="Operation type")
    data: Dict[str, Any] = Field(description="Operation data")
    source_node: str = Field(description="Source node ID")


class DistributedCoordinator:
    """
    Coordinator for distributed vector database cluster.

    Manages:
    - Leader election
    - Replication
    - Health monitoring
    - Failover
    """

    def __init__(
        self,
        node_id: str,
        address: str,
        port: int,
        cluster_nodes: List[str] = None,
        election_timeout_ms: int = 5000,
        heartbeat_interval_ms: int = 1000,
        replication_mode: ReplicationMode = ReplicationMode.ASYNC,
    ):
        """
        Initialize distributed coordinator.

        Args:
            node_id: Unique identifier for this node
            address: Network address for this node
            port: Port for this node
            cluster_nodes: List of other nodes in cluster
            election_timeout_ms: Election timeout in milliseconds
            heartbeat_interval_ms: Heartbeat interval
            replication_mode: Replication mode to use
        """
        self.node_id = node_id
        self.address = address
        self.port = port
        self.cluster_nodes = cluster_nodes or []
        self.election_timeout = election_timeout_ms / 1000
        self.heartbeat_interval = heartbeat_interval_ms / 1000
        self.replication_mode = replication_mode

        # State
        self.role = NodeRole.FOLLOWER
        self.leader_id: Optional[str] = None
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.replication_log: List[ReplicationLog] = []
        self.cluster_state: Dict[str, NodeInfo] = {}

        # Async tasks
        self.heartbeat_task = None
        self.election_task = None
        self.monitor_task = None

    async def start(self):
        """Start distributed coordination."""
        logger.info(f"Starting node {self.node_id} as {self.role}")

        # Initialize self in cluster state
        self.cluster_state[self.node_id] = NodeInfo(
            node_id=self.node_id,
            address=self.address,
            port=self.port,
            role=self.role,
            status=NodeStatus.HEALTHY,
            last_heartbeat=datetime.utcnow(),
        )

        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.election_task = asyncio.create_task(self._election_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop distributed coordination."""
        logger.info(f"Stopping node {self.node_id}")

        # Cancel background tasks
        for task in [self.heartbeat_task, self.election_task, self.monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def handle_write_operation(
        self, operation: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle write operation in distributed system.

        Args:
            operation: Operation type (create, update, delete)
            data: Operation data

        Returns:
            Operation result

        Design choices:
        - Only leader can accept writes
        - Followers forward writes to leader
        - Replication happens after local commit
        """
        if self.role != NodeRole.LEADER:
            # Forward to leader
            if self.leader_id:
                return await self._forward_to_leader(operation, data)
            else:
                raise Exception("No leader available")

        # Leader processes write
        log_entry = ReplicationLog(
            operation=operation, data=data, source_node=self.node_id
        )

        # Add to replication log
        self.replication_log.append(log_entry)

        # Replicate to followers
        if self.replication_mode == ReplicationMode.SYNC:
            await self._replicate_sync(log_entry)
        else:
            asyncio.create_task(self._replicate_async(log_entry))

        return {"status": "success", "log_id": log_entry.log_id}

    async def handle_read_operation(
        self, operation: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle read operation in distributed system.

        Args:
            operation: Read operation type
            data: Query parameters

        Returns:
            Query results

        Design choices:
        - Both leaders and followers can serve reads
        - Optional read-from-leader for strong consistency
        - Load balancing across healthy followers
        """
        # Any node can handle reads
        # In production, might want read preference options
        return {"status": "success", "node": self.node_id, "data": data}

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                if self.role == NodeRole.LEADER:
                    # Leader sends heartbeats to maintain authority
                    await self._send_heartbeats()
                else:
                    # Followers update their own heartbeat
                    if self.node_id in self.cluster_state:
                        self.cluster_state[self.node_id].last_heartbeat = (
                            datetime.utcnow()
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _election_loop(self):
        """Handle leader election."""
        while True:
            try:
                await asyncio.sleep(self.election_timeout)

                if self.role == NodeRole.FOLLOWER:
                    # Check if leader is alive
                    if self.leader_id and self._is_leader_alive():
                        continue

                    # Start election
                    logger.info(f"Node {self.node_id} starting election")
                    await self._start_election()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Election error: {e}")

    async def _monitor_loop(self):
        """Monitor cluster health."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Update cluster state
                await self._update_cluster_state()

                # Check for unhealthy nodes
                unhealthy = [
                    node
                    for node in self.cluster_state.values()
                    if node.status == NodeStatus.UNHEALTHY
                ]

                if unhealthy:
                    logger.warning(
                        f"Unhealthy nodes detected: {[n.node_id for n in unhealthy]}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    async def _start_election(self):
        """Start leader election process (simplified Raft-like)."""
        self.role = NodeRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        votes_received = 1  # Vote for self

        # Request votes from other nodes
        vote_tasks = []
        for node_addr in self.cluster_nodes:
            vote_tasks.append(self._request_vote(node_addr))

        # Wait for votes
        results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict) and result.get("vote_granted"):
                votes_received += 1

        # Check if won election (majority)
        if votes_received > len(self.cluster_nodes) / 2:
            logger.info(f"Node {self.node_id} elected as leader")
            self.role = NodeRole.LEADER
            self.leader_id = self.node_id
            await self._send_heartbeats()  # Establish authority
        else:
            # Failed election, revert to follower
            self.role = NodeRole.FOLLOWER

    async def _request_vote(self, node_addr: str) -> Dict[str, Any]:
        """Request vote from another node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{node_addr}/vote",
                    json={"term": self.current_term, "candidate_id": self.node_id},
                    timeout=2,
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Vote request failed: {e}")
            return {"vote_granted": False}

    async def _send_heartbeats(self):
        """Send heartbeats to all followers."""
        tasks = []
        for node_addr in self.cluster_nodes:
            tasks.append(self._send_heartbeat(node_addr))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_heartbeat(self, node_addr: str):
        """Send heartbeat to a specific node."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{node_addr}/heartbeat",
                    json={
                        "leader_id": self.node_id,
                        "term": self.current_term,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    timeout=1,
                ):
                    pass
        except Exception:
            pass  # Log but don't fail

    async def _replicate_async(self, log_entry: ReplicationLog):
        """Asynchronously replicate to followers."""
        tasks = []
        for node_addr in self.cluster_nodes:
            tasks.append(self._replicate_to_node(node_addr, log_entry))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _replicate_sync(self, log_entry: ReplicationLog):
        """Synchronously replicate to majority of followers."""
        tasks = []
        for node_addr in self.cluster_nodes:
            tasks.append(self._replicate_to_node(node_addr, log_entry))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))

        if successes < len(self.cluster_nodes) / 2:
            raise Exception("Failed to replicate to majority")

    async def _replicate_to_node(
        self, node_addr: str, log_entry: ReplicationLog
    ) -> Dict[str, Any]:
        """Replicate log entry to specific node."""
        try:
        try:
            async with aiohttp.ClientSession() as session:
                payload = (
                    log_entry.model_dump()
                    if hasattr(log_entry, "model_dump")
                    else log_entry.dict()
                )
                ts = payload.get("timestamp")
                if isinstance(ts, datetime):
                    payload["timestamp"] = ts.isoformat()
                async with session.post(
                    f"http://{node_addr}/replicate",
                    json=payload,
                    timeout=5,
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Replication to {node_addr} failed: {e}")
            return {"success": False}

    async def _forward_to_leader(
        self, operation: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward write operation to leader."""
        if not self.leader_id:
            raise Exception("No leader available")

        leader_info = self.cluster_state.get(self.leader_id)
        if not leader_info:
            raise Exception("Leader info not found")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{leader_info.address}:{leader_info.port}/write",
                    json={"operation": operation, "data": data},
                    timeout=10,
                ) as resp:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Failed to forward to leader: {e}")
            raise

    def _is_leader_alive(self) -> bool:
        """Check if current leader is still alive."""
        if not self.leader_id:
            return False

        leader_info = self.cluster_state.get(self.leader_id)
        if not leader_info:
            return False

        # Check last heartbeat time
        time_since_heartbeat = (
            datetime.utcnow() - leader_info.last_heartbeat
        ).total_seconds()
        return time_since_heartbeat < self.election_timeout

    async def _update_cluster_state(self):
        """Update cluster state with node health."""
        for node_id, node_info in self.cluster_state.items():
            time_since_heartbeat = (
                datetime.utcnow() - node_info.last_heartbeat
            ).total_seconds()

            if time_since_heartbeat < self.heartbeat_interval * 2:
                node_info.status = NodeStatus.HEALTHY
            elif time_since_heartbeat < self.heartbeat_interval * 5:
                node_info.status = NodeStatus.DEGRADED
            else:
                node_info.status = NodeStatus.UNHEALTHY

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            "node_id": self.node_id,
            "role": self.role,
            "leader_id": self.leader_id,
            "current_term": self.current_term,
            "cluster_size": len(self.cluster_nodes) + 1,
            "healthy_nodes": sum(
                1
                for n in self.cluster_state.values()
                if n.status == NodeStatus.HEALTHY
            ),
            "replication_mode": self.replication_mode,
            "replication_lag": len(self.replication_log),
            "nodes": [node.dict() for node in self.cluster_state.values()],
        }