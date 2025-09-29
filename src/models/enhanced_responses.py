"""
Enhanced response models with HATEOAS links.

These models extend the base models with hypermedia navigation.
"""

from typing import List, Optional
from uuid import UUID

from pydantic import Field

from models.library import Library as BaseLibrary
from models.document import Document as BaseDocument
from models.chunk import Chunk as BaseChunk
from models.hateoas import HATEOASMixin, Link, HATEOASBuilder


class LibraryResponse(BaseLibrary, HATEOASMixin):
    """Library response with HATEOAS links."""

    @classmethod
    def from_library(cls, library: BaseLibrary, base_url: str = "") -> "LibraryResponse":
        """Create response with HATEOAS links from base library."""
        response = cls(**library.model_dump())
        response.links = HATEOASBuilder.build_library_links(library.id, base_url)
        return response


class DocumentResponse(BaseDocument, HATEOASMixin):
    """Document response with HATEOAS links."""

    @classmethod
    def from_document(cls, document: BaseDocument, base_url: str = "") -> "DocumentResponse":
        """Create response with HATEOAS links from base document."""
        response = cls(**document.model_dump())
        response.links = HATEOASBuilder.build_document_links(
            document.library_id, document.id, base_url
        )
        return response


class ChunkResponse(BaseChunk, HATEOASMixin):
    """Chunk response with HATEOAS links."""

    @classmethod
    def from_chunk(cls, chunk: BaseChunk, document_id: Optional[UUID] = None, base_url: str = "") -> "ChunkResponse":
        """Create response with HATEOAS links from base chunk."""
        response = cls(**chunk.model_dump())
        response.links = HATEOASBuilder.build_chunk_links(
            chunk.library_id, chunk.id, document_id, base_url
        )
        return response


class PaginatedResponseWithLinks(HATEOASMixin):
    """Enhanced paginated response with HATEOAS navigation links."""

    @classmethod
    def create_with_links(
        cls,
        items: List,
        total: int,
        limit: int,
        offset: int,
        base_url: str,
        query_params: Optional[dict] = None
    ):
        """Create paginated response with navigation links."""
        response = cls(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
            has_next=offset + limit < total,
            has_previous=offset > 0
        )

        # Add pagination links
        response.links = HATEOASBuilder.build_pagination_links(
            base_url, offset, limit, total, query_params
        )

        return response