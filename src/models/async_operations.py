"""
Async operation models for long-running tasks.

Following Azure best practices for async operations with 202 status.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OperationStatus(str, Enum):
    """Status of async operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncOperation(BaseModel):
    """Model for tracking async operations."""

    id: UUID = Field(default_factory=uuid4, description="Unique operation ID")
    operation_type: str = Field(description="Type of operation (e.g., 'index_build')")
    status: OperationStatus = Field(default=OperationStatus.PENDING)
    resource_id: UUID = Field(description="ID of the resource being operated on")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress as fraction 0-1")
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    @property
    def is_terminal(self) -> bool:
        """Check if operation is in a terminal state."""
        return self.status in [OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class AsyncOperationResponse(BaseModel):
    """Response model for async operation status."""

    operation_id: UUID = Field(description="Unique operation ID")
    status: OperationStatus = Field(description="Current operation status")
    operation_type: str = Field(description="Type of operation")
    resource_id: UUID = Field(description="ID of the resource being operated on")
    created_at: datetime = Field(description="When operation was created")
    started_at: Optional[datetime] = Field(None, description="When operation started")
    completed_at: Optional[datetime] = Field(None, description="When operation completed")
    progress: Optional[float] = Field(None, description="Progress as fraction 0-1")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result if completed")
    links: Optional[Dict[str, str]] = Field(None, description="Related resource links")


class AsyncOperationManager:
    """Manager for async operations."""

    def __init__(self):
        """Initialize with empty operation store."""
        self._operations: Dict[UUID, AsyncOperation] = {}
        self._lock = asyncio.Lock()

    async def create_operation(
        self,
        operation_type: str,
        resource_id: UUID
    ) -> AsyncOperation:
        """
        Create a new async operation.

        Args:
            operation_type: Type of operation
            resource_id: ID of resource being operated on

        Returns:
            Created operation
        """
        async with self._lock:
            operation = AsyncOperation(
                operation_type=operation_type,
                resource_id=resource_id
            )
            self._operations[operation.id] = operation
            return operation

    async def get_operation(self, operation_id: UUID) -> Optional[AsyncOperation]:
        """Get operation by ID."""
        async with self._lock:
            return self._operations.get(operation_id)

    async def update_operation(
        self,
        operation_id: UUID,
        **updates
    ) -> Optional[AsyncOperation]:
        """
        Update operation fields.

        Args:
            operation_id: Operation ID
            **updates: Fields to update

        Returns:
            Updated operation or None if not found
        """
        async with self._lock:
            if operation_id not in self._operations:
                return None

            operation = self._operations[operation_id]
            for field, value in updates.items():
                if hasattr(operation, field):
                    setattr(operation, field, value)

            return operation

    async def start_operation(self, operation_id: UUID) -> Optional[AsyncOperation]:
        """Mark operation as started."""
        return await self.update_operation(
            operation_id,
            status=OperationStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

    async def complete_operation(
        self,
        operation_id: UUID,
        result: Optional[Dict[str, Any]] = None
    ) -> Optional[AsyncOperation]:
        """Mark operation as completed."""
        return await self.update_operation(
            operation_id,
            status=OperationStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            result=result
        )

    async def fail_operation(
        self,
        operation_id: UUID,
        error_message: str
    ) -> Optional[AsyncOperation]:
        """Mark operation as failed."""
        return await self.update_operation(
            operation_id,
            status=OperationStatus.FAILED,
            completed_at=datetime.utcnow(),
            error_message=error_message
        )

    async def cleanup_old_operations(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed operations.

        Args:
            max_age_hours: Maximum age of operations to keep

        Returns:
            Number of operations cleaned up
        """
        async with self._lock:
            cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
            to_remove = []

            for op_id, operation in self._operations.items():
                if (operation.is_terminal and
                    operation.completed_at and
                    operation.completed_at.timestamp() < cutoff):
                    to_remove.append(op_id)

            for op_id in to_remove:
                del self._operations[op_id]

            return len(to_remove)


# Global operation manager instance
operation_manager = AsyncOperationManager()