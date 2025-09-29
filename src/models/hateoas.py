"""
HATEOAS (Hypermedia as the Engine of Application State) support.

Provides navigation links in API responses following RESTful principles.
"""

from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field


class Link(BaseModel):
    """Represents a hypermedia link."""

    rel: str = Field(description="Relationship type (self, edit, delete, etc.)")
    href: str = Field(description="URL to the linked resource")
    method: str = Field(default="GET", description="HTTP method for the link")
    type: Optional[str] = Field(None, description="Media type of the linked resource")
    title: Optional[str] = Field(None, description="Human-readable title")

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class HATEOASMixin(BaseModel):
    """Mixin to add HATEOAS links to response models."""

    links: List[Link] = Field(default_factory=list, description="Hypermedia navigation links")

    def add_link(self, rel: str, href: str, method: str = "GET", type: Optional[str] = None, title: Optional[str] = None):
        """Add a hypermedia link to this resource."""
        link = Link(rel=rel, href=href, method=method, type=type, title=title)
        self.links.append(link)

    def get_link(self, rel: str) -> Optional[Link]:
        """Get a link by relationship type."""
        for link in self.links:
            if link.rel == rel:
                return link
        return None


class HATEOASBuilder:
    """Builder for creating HATEOAS links."""

    @staticmethod
    def build_library_links(library_id: UUID, base_url: str = "") -> List[Link]:
        """Build standard links for a library resource."""
        return [
            Link(rel="self", href=f"{base_url}/libraries/{library_id}", method="GET"),
            Link(rel="edit", href=f"{base_url}/libraries/{library_id}", method="PUT"),
            Link(rel="delete", href=f"{base_url}/libraries/{library_id}", method="DELETE"),
            Link(rel="documents", href=f"{base_url}/libraries/{library_id}/documents", method="GET"),
            Link(rel="chunks", href=f"{base_url}/libraries/{library_id}/chunks", method="GET"),
            Link(rel="index", href=f"{base_url}/libraries/{library_id}/index", method="POST", title="Build index"),
            Link(rel="index-stats", href=f"{base_url}/libraries/{library_id}/index/stats", method="GET"),
            Link(rel="search", href=f"{base_url}/libraries/{library_id}/search", method="GET"),
        ]

    @staticmethod
    def build_document_links(library_id: UUID, document_id: UUID, base_url: str = "") -> List[Link]:
        """Build standard links for a document resource."""
        return [
            Link(rel="self", href=f"{base_url}/libraries/{library_id}/documents/{document_id}", method="GET"),
            Link(rel="edit", href=f"{base_url}/libraries/{library_id}/documents/{document_id}", method="PUT"),
            Link(rel="delete", href=f"{base_url}/libraries/{library_id}/documents/{document_id}", method="DELETE"),
            Link(rel="library", href=f"{base_url}/libraries/{library_id}", method="GET"),
            Link(rel="chunks", href=f"{base_url}/libraries/{library_id}/chunks?document_id={document_id}", method="GET"),
        ]

    @staticmethod
    def build_chunk_links(library_id: UUID, chunk_id: UUID, document_id: Optional[UUID] = None, base_url: str = "") -> List[Link]:
        """Build standard links for a chunk resource."""
        links = [
            Link(rel="self", href=f"{base_url}/libraries/{library_id}/chunks/{chunk_id}", method="GET"),
            Link(rel="edit", href=f"{base_url}/libraries/{library_id}/chunks/{chunk_id}", method="PUT"),
            Link(rel="delete", href=f"{base_url}/libraries/{library_id}/chunks/{chunk_id}", method="DELETE"),
            Link(rel="library", href=f"{base_url}/libraries/{library_id}", method="GET"),
        ]

        if document_id:
            links.append(
                Link(rel="document", href=f"{base_url}/libraries/{library_id}/documents/{document_id}", method="GET")
            )

        return links

    @staticmethod
    def build_search_links(library_id: UUID, base_url: str = "") -> List[Link]:
        """Build links for search operations."""
        return [
            Link(rel="self", href=f"{base_url}/libraries/{library_id}/search", method="GET"),
            Link(rel="library", href=f"{base_url}/libraries/{library_id}", method="GET"),
            Link(rel="post-search", href=f"{base_url}/libraries/{library_id}/search", method="POST", title="Advanced search"),
        ]

    @staticmethod
    def build_pagination_links(
        base_url: str,
        current_offset: int,
        limit: int,
        total: int,
        query_params: Optional[Dict[str, Any]] = None
    ) -> List[Link]:
        """Build pagination navigation links."""
        links = []
        params = query_params or {}

        # Build query string
        def build_url(offset: int) -> str:
            params_copy = params.copy()
            params_copy.update({"limit": limit, "offset": offset})
            query_string = "&".join(f"{k}={v}" for k, v in params_copy.items())
            return f"{base_url}?{query_string}"

        # Self link
        links.append(Link(rel="self", href=build_url(current_offset), method="GET"))

        # First page
        if current_offset > 0:
            links.append(Link(rel="first", href=build_url(0), method="GET"))

        # Previous page
        if current_offset > 0:
            prev_offset = max(0, current_offset - limit)
            links.append(Link(rel="prev", href=build_url(prev_offset), method="GET"))

        # Next page
        if current_offset + limit < total:
            next_offset = current_offset + limit
            links.append(Link(rel="next", href=build_url(next_offset), method="GET"))

        # Last page
        if current_offset + limit < total:
            last_offset = ((total - 1) // limit) * limit
            links.append(Link(rel="last", href=build_url(last_offset), method="GET"))

        return links

    @staticmethod
    def build_operation_links(operation_id: UUID, resource_type: str, resource_id: UUID, base_url: str = "") -> List[Link]:
        """Build links for async operations."""
        links = [
            Link(rel="self", href=f"{base_url}/api/operations/{operation_id}", method="GET"),
            Link(rel="cancel", href=f"{base_url}/api/operations/{operation_id}", method="DELETE", title="Cancel operation"),
        ]

        # Add resource-specific links
        if resource_type == "library":
            links.append(Link(rel="resource", href=f"{base_url}/libraries/{resource_id}", method="GET"))
        elif resource_type == "document":
            # Would need library_id for full path
            pass
        elif resource_type == "chunk":
            # Would need library_id for full path
            pass

        return links