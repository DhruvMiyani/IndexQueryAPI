"""
Search API endpoints.

Handles kNN vector search operations.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query

from models.search import SearchRequest, SearchResponse
from services.search_service import SearchService
from repository.base import NotFoundError
from .dependencies import get_search_service

router = APIRouter(prefix="/libraries/{library_id}", tags=["search"])




@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Perform vector similarity search",
)
async def search_library(
    library_id: UUID,
    top_k: int = Query(5, ge=1, le=100, description="Number of results to return"),
    query_text: Optional[str] = Query(None, description="Text query to embed and search"),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity score"),
    include_text: bool = Query(True, description="Include chunk text in results"),
    include_metadata: bool = Query(True, description="Include metadata in results"),
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Perform k-nearest neighbor vector search in a library.

    Provide either query_text (which will be embedded) or query_vector in request body.

    - **library_id**: ID of the library to search
    - **top_k**: Number of most similar results to return (1-100)
    - **query_text**: Text to embed and search for (alternative to query_vector)
    - **min_score**: Minimum similarity score threshold (0.0-1.0)
    - **include_text**: Whether to include chunk text in results
    - **include_metadata**: Whether to include chunk and document metadata
    """
    try:
        # Create search request
        search_request = SearchRequest(
            query_text=query_text,
            top_k=top_k,
            min_score=min_score,
            include_text=include_text,
            include_metadata=include_metadata,
        )

        return await search_service.search(library_id, search_request)
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
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Perform vector search with request body",
)
async def search_library_post(
    library_id: UUID,
    search_request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Perform k-nearest neighbor vector search with full request body.

    This endpoint allows specifying query_vector, filters, and all search parameters
    in the request body for more complex queries.

    - **library_id**: ID of the library to search
    - **search_request**: Complete search parameters including query vector or text
    """
    try:
        return await search_service.search(library_id, search_request)
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
            detail=f"Search failed: {str(e)}"
        )