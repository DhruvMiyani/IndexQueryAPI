"""
Operations API endpoints for tracking async operations.

Provides status monitoring for long-running operations.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from models.async_operations import AsyncOperationResponse, operation_manager

router = APIRouter(prefix="/api/operations", tags=["operations"])


@router.get(
    "/{operation_id}",
    response_model=AsyncOperationResponse,
    summary="Get operation status",
)
async def get_operation_status(operation_id: UUID) -> AsyncOperationResponse:
    """
    Get the status of an async operation.

    Use this endpoint to poll the status of long-running operations
    like index building.

    - **operation_id**: ID of the operation to check

    Returns:
    - **200 OK**: Operation is still running or completed
    - **303 See Other**: Operation completed, check Location header for result
    - **404 Not Found**: Operation not found
    """
    operation = await operation_manager.get_operation(operation_id)

    if not operation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Operation {operation_id} not found"
        )

    response = AsyncOperationResponse(
        operation_id=operation.id,
        status=operation.status,
        operation_type=operation.operation_type,
        resource_id=operation.resource_id,
        created_at=operation.created_at,
        started_at=operation.started_at,
        completed_at=operation.completed_at,
        progress=operation.progress,
        error_message=operation.error_message,
        result=operation.result,
        links={
            "self": f"/api/operations/{operation.id}",
        }
    )

    # Add resource-specific links
    if operation.operation_type == "index_build":
        response.links.update({
            "resource": f"/libraries/{operation.resource_id}",
            "index_stats": f"/libraries/{operation.resource_id}/index/stats"
        })

    return response


@router.delete(
    "/{operation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel operation",
)
async def cancel_operation(operation_id: UUID):
    """
    Cancel a running operation.

    Note: Some operations may not be immediately cancellable
    if they are already in progress.

    - **operation_id**: ID of the operation to cancel
    """
    operation = await operation_manager.get_operation(operation_id)

    if not operation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Operation {operation_id} not found"
        )

    if operation.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Operation {operation_id} is already {operation.status.value}"
        )

    # Mark as cancelled (actual cancellation logic would depend on operation type)
    await operation_manager.update_operation(
        operation_id,
        status="cancelled",
        completed_at=operation.created_at.__class__.utcnow()
    )

    return None