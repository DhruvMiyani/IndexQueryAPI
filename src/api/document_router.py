"""
Document API endpoints.

Handles CRUD operations for documents within libraries.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import Response

from models.document import Document, DocumentCreate, DocumentUpdate
from models.pagination import PaginatedResponse
from models.field_selection import FieldSelector
from services.document_service import DocumentService
from repository.base import NotFoundError
from .dependencies import get_document_service

router = APIRouter(prefix="/libraries/{library_id}/documents", tags=["documents"])


@router.post(
    "/",
    response_model=Document,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new document",
)
async def create_document(
    library_id: UUID,
    document_data: DocumentCreate,
    chunk_texts: Optional[List[str]] = None,
    document_service: DocumentService = Depends(get_document_service),
) -> Document:
    """
    Create a new document in the specified library.

    Can optionally include chunk texts that will be automatically embedded.

    - **library_id**: ID of the library to create the document in
    - **document_data**: Document metadata
    - **chunk_texts**: Optional list of chunk texts to embed and attach
    """
    try:
        return await document_service.create_document(
            library_id, document_data, chunk_texts
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}"
        )


@router.get(
    "/",
    response_model=PaginatedResponse[Document],
    summary="List documents in library",
)
async def list_documents(
    library_id: UUID,
    limit: int = Query(25, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include"),
    document_service: DocumentService = Depends(get_document_service),
) -> PaginatedResponse[Document]:
    """
    List all documents in the specified library with pagination.

    - **library_id**: ID of the library
    - **limit**: Maximum number of documents to return (1-100, default 25)
    - **offset**: Number of documents to skip (default 0)
    """
    try:
        documents, total = await document_service.list_documents_paginated(
            library_id, offset=offset, limit=limit
        )

        # Apply field selection if requested
        field_set = FieldSelector.parse_fields_query(fields)
        if field_set:
            FieldSelector.validate_fields(field_set, Document)
            filtered_documents = [
                Document.model_validate(FieldSelector.select_fields(document, field_set))
                for document in documents
            ]
        else:
            filtered_documents = documents

        return PaginatedResponse(
            items=filtered_documents,
            total=total,
            limit=limit,
            offset=offset,
            has_next=offset + limit < total,
            has_previous=offset > 0
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {library_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get(
    "/{document_id}",
    summary="Get document with chunks",
)
async def get_document(
    library_id: UUID,
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> dict:
    """
    Get a specific document and its chunks.

    Returns both document metadata and associated chunk data.
    """
    try:
        return await document_service.get_document_with_chunks(document_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.put(
    "/{document_id}",
    response_model=Document,
    summary="Update document metadata",
)
async def update_document(
    library_id: UUID,
    document_id: UUID,
    document_data: DocumentUpdate,
    document_service: DocumentService = Depends(get_document_service),
) -> Document:
    """
    Update document metadata.

    Only provided fields will be updated.
    """
    try:
        return await document_service.update_document(document_id, document_data)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
)
async def delete_document(
    library_id: UUID,
    document_id: UUID,
    document_service: DocumentService = Depends(get_document_service),
) -> Response:
    """
    Delete a document and all its chunks.

    This operation is irreversible.
    """
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )