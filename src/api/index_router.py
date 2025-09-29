"""
Index API endpoints.

Handles vector indexing operations for libraries.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Response
from pydantic import BaseModel, Field

from services.index_service import IndexService
from indexes.base import IndexType
from repository.base import NotFoundError
from models.async_operations import AsyncOperationResponse, operation_manager
from .dependencies import get_index_service

router = APIRouter(prefix="/libraries/{library_id}", tags=["indexing"])


class IndexRequest(BaseModel):
    """Request model for index building."""
    index_type: Optional[IndexType] = Field(
        None,
        description="Type of index to build. Valid options: 'linear', 'kd_tree', 'lsh', 'optimized_linear', 'improved_kd_tree', 'multiprobe_lsh', 'hnsw', 'ivf_pq'",
        example="optimized_linear"
    )
    force_rebuild: bool = Field(
        False,
        description="Force rebuild even if index already exists"
    )
    async_operation: bool = Field(
        False,
        description="If true, returns 202 Accepted and runs in background"
    )


class IndexResponse(BaseModel):
    """Response model for index operations."""
    message: str
    index_type: str  # Keep as string for display purposes
    library_id: UUID
    chunk_count: int


@router.post(
    "/index",
    summary="Build or update library index",
)
async def build_index(
    library_id: UUID,
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    index_service: IndexService = Depends(get_index_service),
):
    """
    Build or rebuild the vector index for a library.

    - **library_id**: ID of the library to index
    - **index_type**: Type of index to build. Options:
        - Original: linear, kd_tree, lsh
        - Optimized: optimized_linear (95x faster), improved_kd_tree (better metrics), multiprobe_lsh (120% better recall)
        - Advanced: hnsw (state-of-the-art), ivf_pq (memory efficient)
        Auto-selected if not provided
    - **force_rebuild**: Force rebuild even if index already exists
    - **async_operation**: If true, returns 202 Accepted and runs in background

    Returns 201 Created with IndexResponse for sync operations,
    or 202 Accepted with operation tracking for async operations.
    """
    try:
        # Use the enum directly from request
        index_type = request.index_type

        # Check if async operation is requested
        if request.async_operation:
            # Create async operation
            operation = await operation_manager.create_operation(
                operation_type="index_build",
                resource_id=library_id
            )

            # Start background task
            background_tasks.add_task(
                _build_index_async,
                operation.id,
                library_id,
                index_type,
                request.force_rebuild,
                index_service
            )

            # Return 202 Accepted with operation details
            response = Response(
                status_code=status.HTTP_202_ACCEPTED,
                headers={
                    "Location": f"/api/operations/{operation.id}"
                }
            )

            # Return operation details in response body
            operation_response = AsyncOperationResponse(
                operation_id=operation.id,
                status=operation.status,
                operation_type=operation.operation_type,
                resource_id=library_id,
                created_at=operation.created_at,
                links={
                    "status": f"/api/operations/{operation.id}",
                    "cancel": f"/api/operations/{operation.id}"
                }
            )
            response.body = operation_response.model_dump_json().encode()
            response.media_type = "application/json"
            return response
        else:
            # Synchronous operation
            index = await index_service.build_index(
                library_id, index_type, request.force_rebuild
            )

            # Get library for chunk count
            from .dependencies import get_library_service
            library_service = get_library_service()
            library = await library_service.get_library(library_id)

            return IndexResponse(
                message=f"Index built successfully for library {library_id}",
                index_type=index_type.value if index_type else "auto-selected",
                library_id=library_id,
                chunk_count=library.chunk_count,
            )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build index: {str(e)}"
        )


@router.get(
    "/index/stats",
    summary="Get index statistics",
)
async def get_index_stats(
    library_id: UUID,
    index_service: IndexService = Depends(get_index_service),
) -> dict:
    """
    Get statistics and information about the library's index.

    Returns index type, status, and performance metrics if available.
    """
    try:
        return await index_service.get_index_stats(library_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index stats: {str(e)}"
        )


@router.delete(
    "/index",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete library index",
)
async def delete_index(
    library_id: UUID,
    index_service: IndexService = Depends(get_index_service),
):
    """
    Delete the vector index for a library.

    This will fall back to linear search for queries until rebuilt.
    """
    try:
        success = await index_service.delete_index(library_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No index found for library {library_id}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete index: {str(e)}"
        )


async def _build_index_async(
    operation_id: UUID,
    library_id: UUID,
    index_type: Optional[IndexType],
    force_rebuild: bool,
    index_service: IndexService
):
    """
    Background task for async index building.

    Args:
        operation_id: ID of the async operation
        library_id: Library to index
        index_type: Type of index to build
        force_rebuild: Whether to force rebuild
        index_service: Index service instance
    """
    try:
        # Mark operation as started
        await operation_manager.start_operation(operation_id)

        # Build the index
        index = await index_service.build_index(
            library_id, index_type, force_rebuild
        )

        # Mark operation as completed with result
        await operation_manager.complete_operation(
            operation_id,
            result={
                "index_type": index_type.value if index_type else "auto-selected",
                "library_id": str(library_id),
                "message": "Index built successfully"
            }
        )
    except Exception as e:
        # Mark operation as failed
        await operation_manager.fail_operation(
            operation_id,
            error_message=str(e)
        )