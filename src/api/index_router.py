"""
Index API endpoints.

Handles vector indexing operations for libraries.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from services.index_service import IndexService
from indexes import IndexType
from repository.base import NotFoundError
from .dependencies import get_index_service

router = APIRouter(prefix="/libraries/{library_id}", tags=["indexing"])


class IndexRequest(BaseModel):
    """Request model for index building."""
    index_type: Optional[str] = None
    force_rebuild: bool = False


class IndexResponse(BaseModel):
    """Response model for index operations."""
    message: str
    index_type: str
    library_id: UUID
    chunk_count: int


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Build or update library index",
)
async def build_index(
    library_id: UUID,
    request: IndexRequest,
    index_service: IndexService = Depends(get_index_service),
) -> IndexResponse:
    """
    Build or rebuild the vector index for a library.

    - **library_id**: ID of the library to index
    - **index_type**: Type of index to build (linear, kd_tree, lsh). Auto-selected if not provided
    - **force_rebuild**: Force rebuild even if index already exists
    """
    try:
        # Convert string to IndexType enum if provided
        index_type = None
        if request.index_type:
            try:
                index_type = IndexType(request.index_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid index type: {request.index_type}. Valid types: linear, kd_tree, lsh"
                )

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