"""
Chunk API endpoints.

Handles CRUD operations for chunks within libraries.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from models.chunk import Chunk, ChunkCreate, ChunkUpdate
from services.chunk_service import ChunkService
from repository.base import NotFoundError
from .dependencies import get_chunk_service

router = APIRouter(prefix="/libraries/{library_id}/chunks", tags=["chunks"])


@router.post(
    "/",
    response_model=Chunk,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chunk",
)
async def create_chunk(
    library_id: UUID,
    chunk_data: ChunkCreate,
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> Chunk:
    """
    Create a new chunk in the specified library.

    If no embedding is provided, one will be generated from the text.

    - **library_id**: ID of the library (used for indexing)
    - **chunk_data**: Chunk data including text, document_id, and optional embedding
    """
    try:
        return await chunk_service.create_chunk(chunk_data.document_id, chunk_data, library_id)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chunk: {str(e)}"
        )


@router.post(
    "/bulk",
    response_model=List[Chunk],
    status_code=status.HTTP_201_CREATED,
    summary="Create multiple chunks",
)
async def create_chunks_bulk(
    library_id: UUID,
    document_id: UUID,
    texts: List[str],
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> List[Chunk]:
    """
    Create multiple chunks at once from a list of texts.

    Embeddings will be generated for all texts in batch.

    - **library_id**: ID of the library
    - **document_id**: ID of the parent document
    - **texts**: List of chunk texts to create
    """
    try:
        return await chunk_service.bulk_create_chunks(document_id, texts, library_id)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chunks: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[Chunk],
    summary="List chunks in library",
)
async def list_chunks(
    library_id: UUID,
    skip: int = 0,
    limit: int = 100,
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> List[Chunk]:
    """
    List all chunks in the specified library.

    - **library_id**: ID of the library
    - **skip**: Number of chunks to skip (for pagination)
    - **limit**: Maximum number of chunks to return
    """
    try:
        return await chunk_service.list_chunks_by_library(
            library_id, offset=skip, limit=limit
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list chunks: {str(e)}"
        )


@router.get(
    "/{chunk_id}",
    response_model=Chunk,
    summary="Get chunk details",
)
async def get_chunk(
    library_id: UUID,
    chunk_id: UUID,
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> Chunk:
    """
    Get a specific chunk by ID.

    Returns chunk content, metadata, and embedding.
    """
    try:
        return await chunk_service.get_chunk(chunk_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunk: {str(e)}"
        )


@router.put(
    "/{chunk_id}",
    response_model=Chunk,
    summary="Update chunk",
)
async def update_chunk(
    library_id: UUID,
    chunk_id: UUID,
    chunk_data: ChunkUpdate,
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> Chunk:
    """
    Update chunk text or metadata.

    If text is changed, the embedding will be automatically regenerated.
    """
    try:
        return await chunk_service.update_chunk(chunk_id, chunk_data)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update chunk: {str(e)}"
        )


@router.delete(
    "/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chunk",
)
async def delete_chunk(
    library_id: UUID,
    chunk_id: UUID,
    chunk_service: ChunkService = Depends(get_chunk_service),
) -> Response:
    """
    Delete a chunk.

    This operation is irreversible.
    """
    try:
        success = await chunk_service.delete_chunk(chunk_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chunk: {str(e)}"
        )