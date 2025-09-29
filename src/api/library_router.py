"""
Library API endpoints following Clean Code principles.

Handles CRUD operations for libraries with meaningful names,
early returns, and single responsibility functions.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import Response

from models.library import Library, LibraryCreate, LibraryUpdate
from models.pagination import PaginatedResponse, PaginationParams
from models.field_selection import FieldSelector
from models.enhanced_responses import LibraryResponse
from services.library_service import LibraryService
from repository.base import NotFoundError, DuplicateError
from core.constants import (
    DEFAULT_PAGE_LIMIT,
    DEFAULT_PAGE_OFFSET,
    ERROR_LIBRARY_NOT_FOUND,
    ERROR_LIBRARY_CREATION_FAILED,
)
from .dependencies import get_library_service

router = APIRouter(prefix="/libraries", tags=["libraries"])


def handle_library_not_found_error(library_id: UUID) -> HTTPException:
    """Create HTTP 404 exception for library not found."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{ERROR_LIBRARY_NOT_FOUND}: {library_id}"
    )


def handle_duplicate_library_error(error: DuplicateError) -> HTTPException:
    """Create HTTP 400 exception for duplicate library."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error)
    )


def handle_internal_server_error(operation: str, error: Exception) -> HTTPException:
    """Create HTTP 500 exception for internal server errors."""
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to {operation}: {str(error)}"
    )


def validate_pagination_parameters(skip: int, limit: int) -> None:
    """Validate pagination parameters and raise HTTP exception if invalid."""
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )

    if limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be positive"
        )


@router.post(
    "/",
    response_model=Library,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new library",
)
async def create_library_endpoint(
    library_creation_data: LibraryCreate,
    library_service: LibraryService = Depends(get_library_service),
) -> Library:
    """
    Create a new library with the provided metadata.

    Returns the created library with generated ID.
    Follows Clean Code: early returns, meaningful names.
    """
    try:
        return await library_service.create_library(library_creation_data)
    except DuplicateError as duplicate_error:
        raise handle_duplicate_library_error(duplicate_error)
    except Exception as unexpected_error:
        raise handle_internal_server_error("create library", unexpected_error)


@router.get(
    "/",
    response_model=PaginatedResponse[LibraryResponse],
    summary="List all libraries",
)
async def list_libraries_endpoint(
    limit: int = Query(25, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
    library_service: LibraryService = Depends(get_library_service),
) -> PaginatedResponse[LibraryResponse]:
    """
    List all libraries with standardized pagination.

    - **limit**: Maximum number of libraries to return (1-100, default 25)
    - **offset**: Number of libraries to skip (default 0)
    """
    try:
        libraries, total = await library_service.list_libraries_paginated(
            offset=offset,
            limit=limit
        )

        # Convert to HATEOAS responses
        library_responses = [
            LibraryResponse.from_library(library)
            for library in libraries
        ]

        # Apply field selection if requested
        field_set = FieldSelector.parse_fields_query(fields)
        if field_set:
            FieldSelector.validate_fields(field_set, LibraryResponse)
            filtered_libraries = [
                LibraryResponse.model_validate(FieldSelector.select_fields(lib_resp, field_set))
                for lib_resp in library_responses
            ]
        else:
            filtered_libraries = library_responses

        return PaginatedResponse(
            items=filtered_libraries,
            total=total,
            limit=limit,
            offset=offset,
            has_next=offset + limit < total,
            has_previous=offset > 0
        )
    except Exception as unexpected_error:
        raise handle_internal_server_error("list libraries", unexpected_error)


@router.get(
    "/{library_id}",
    response_model=Library,
    summary="Get library details",
)
async def get_library_by_id_endpoint(
    library_id: UUID,
    library_service: LibraryService = Depends(get_library_service),
) -> Library:
    """
    Get details of a specific library by ID.

    Returns library information including document and chunk counts.
    Early return on not found for clean code.
    """
    try:
        return await library_service.get_library(library_id)
    except NotFoundError:
        raise handle_library_not_found_error(library_id)
    except Exception as unexpected_error:
        raise handle_internal_server_error("get library", unexpected_error)


@router.put(
    "/{library_id}",
    response_model=Library,
    summary="Update library metadata",
)
async def update_library_metadata_endpoint(
    library_id: UUID,
    library_update_data: LibraryUpdate,
    library_service: LibraryService = Depends(get_library_service),
) -> Library:
    """
    Update library metadata.

    Only provided fields will be updated.
    Follows clean code with meaningful names and early returns.
    """
    try:
        return await library_service.update_library(library_id, library_update_data)
    except NotFoundError:
        raise handle_library_not_found_error(library_id)
    except DuplicateError as duplicate_error:
        raise handle_duplicate_library_error(duplicate_error)
    except Exception as unexpected_error:
        raise handle_internal_server_error("update library", unexpected_error)


@router.delete(
    "/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete library",
)
async def delete_library_endpoint(
    library_id: UUID,
    library_service: LibraryService = Depends(get_library_service),
) -> Response:
    """
    Delete a library and all its contents (documents and chunks).

    This operation is irreversible.
    Early return pattern for not found case.
    """
    try:
        deletion_successful = await library_service.delete_library(library_id)
        if not deletion_successful:
            raise handle_library_not_found_error(library_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as unexpected_error:
        raise handle_internal_server_error("delete library", unexpected_error)


@router.get(
    "/{library_id}/statistics",
    summary="Get library statistics",
)
async def get_library_statistics_endpoint(
    library_id: UUID,
    library_service: LibraryService = Depends(get_library_service),
) -> dict:
    """
    Get detailed statistics for a library.

    Returns document count, chunk count, average chunks per document, etc.
    Clean code implementation with meaningful names.
    """
    try:
        return await library_service.get_library_statistics(library_id)
    except NotFoundError:
        raise handle_library_not_found_error(library_id)
    except Exception as unexpected_error:
        raise handle_internal_server_error("get library statistics", unexpected_error)