"""
Python SDK Client for Vector Database API.

Provides a high-level, pythonic interface for interacting with the vector database.
Includes comprehensive documentation and type hints for excellent developer experience.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import logging
from enum import Enum

import requests
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Available index types."""

    LINEAR = "linear"
    KD_TREE = "kd_tree"
    LSH = "lsh"
    OPTIMIZED_LINEAR = "optimized_linear"
    IMPROVED_KD_TREE = "improved_kd_tree"
    MULTIPROBE_LSH = "multiprobe_lsh"
    HNSW = "hnsw"
    IVF_PQ = "ivf_pq"


class Library(BaseModel):
    """Library model."""

    id: UUID
    name: str
    document_ids: List[UUID] = Field(default_factory=list)
    chunk_count: int = 0
    document_count: int = 0
    index_type: Optional[str] = None
    index_status: str = "none"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def created_at(self) -> Optional[datetime]:
        """Get created_at from metadata."""
        return self.metadata.get("created_at")

    @property
    def updated_at(self) -> Optional[datetime]:
        """Get updated_at from metadata."""
        return self.metadata.get("updated_at")

    @property
    def description(self) -> Optional[str]:
        """Get description from metadata."""
        return self.metadata.get("description")


class Document(BaseModel):
    """Document model."""

    id: UUID
    library_id: UUID
    title: Optional[str] = None
    content: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def created_at(self) -> Optional[datetime]:
        """Get created_at from metadata."""
        return self.metadata.get("created_at")

    @property
    def updated_at(self) -> Optional[datetime]:
        """Get updated_at from metadata."""
        return self.metadata.get("updated_at")


class Chunk(BaseModel):
    """Chunk model."""

    id: UUID
    document_id: UUID
    library_id: UUID
    text: str
    embedding: Optional[List[float]] = None
    position: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result model."""

    chunk_id: UUID
    document_id: UUID
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict] = None
    document_metadata: Optional[Dict] = None


class VectorDBClient:
    """
    Python client for Vector Database API.

    Example usage:
        ```python
        from vectordb_client import VectorDBClient

        # Initialize client
        client = VectorDBClient("http://localhost:8000")

        # Create a library
        library = client.create_library(
            name="Research Papers",
            description="Collection of AI research papers"
        )

        # Add documents
        doc = client.create_document(
            library_id=library.id,
            title="Introduction to Machine Learning",
            chunks=[
                {"text": "Machine learning is...", "embedding": [0.1, 0.2, ...]},
                {"text": "Neural networks are...", "embedding": [0.3, 0.4, ...]}
            ]
        )

        # Build index
        client.build_index(library.id, index_type=IndexType.HNSW)

        # Search
        results = client.search(
            library_id=library.id,
            query_text="What is deep learning?",
            top_k=5
        )

        for result in results:
            print(f"Score: {result.score:.3f}, Text: {result.text[:100]}...")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize Vector Database client.

        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        # Set headers
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    # Library operations

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Library:
        """
        Create a new library.

        Args:
            name: Library name
            description: Optional description
            metadata: Optional metadata dictionary

        Returns:
            Created library object

        Example:
            ```python
            library = client.create_library(
                name="Product Reviews",
                description="Customer product reviews",
                metadata={"category": "e-commerce", "language": "en"}
            )
            ```
        """
        data = {"name": name}
        if description:
            data["description"] = description
        if metadata:
            data["metadata"] = metadata

        response = self._request("POST", "/libraries", json=data)
        return Library(**response)

    def get_library(self, library_id: Union[str, UUID]) -> Library:
        """
        Get library by ID.

        Args:
            library_id: Library UUID

        Returns:
            Library object
        """
        response = self._request("GET", f"/libraries/{library_id}")
        return Library(**response)

    def list_libraries(
        self, limit: int = 100, offset: int = 0
    ) -> List[Library]:
        """
        List all libraries.

        Args:
            limit: Maximum number of libraries to return
            offset: Number of libraries to skip

        Returns:
            List of library objects
        """
        params = {"limit": limit, "offset": offset}
        response = self._request("GET", "/libraries", params=params)
        return [Library(**lib) for lib in response.get("items", [])]

    def update_library(
        self,
        library_id: Union[str, UUID],
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Library:
        """
        Update library.

        Args:
            library_id: Library UUID
            name: New name
            description: New description
            metadata: New metadata

        Returns:
            Updated library object
        """
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if metadata is not None:
            data["metadata"] = metadata

        response = self._request("PATCH", f"/libraries/{library_id}", json=data)
        return Library(**response)

    def delete_library(self, library_id: Union[str, UUID]) -> bool:
        """
        Delete library.

        Args:
            library_id: Library UUID

        Returns:
            True if deleted successfully
        """
        self._request("DELETE", f"/libraries/{library_id}")
        return True

    # Document operations

    def create_document(
        self,
        library_id: Union[str, UUID],
        title: Optional[str] = None,
        content: Optional[str] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Create a document with chunks.

        Args:
            library_id: Library UUID
            title: Document title
            content: Document content
            chunks: List of chunk dictionaries with 'text' and optional 'embedding'
            metadata: Document metadata

        Returns:
            Created document object

        Example:
            ```python
            doc = client.create_document(
                library_id=library.id,
                title="Chapter 1",
                chunks=[
                    {"text": "First paragraph...", "embedding": [...]},
                    {"text": "Second paragraph...", "embedding": [...]}
                ],
                metadata={"author": "John Doe", "year": 2023}
            )
            ```
        """
        # Build document metadata with required title field
        doc_metadata = metadata.copy() if metadata else {}
        if title:
            doc_metadata["title"] = title
        elif "title" not in doc_metadata:
            doc_metadata["title"] = "Untitled Document"

        title_value = title or doc_metadata.get("title") or "Untitled Document"
        document_data: Dict[str, Any] = {"title": title_value, "metadata": doc_metadata}
        if content:
            document_data["content"] = content

        payload: Dict[str, Any] = {"document_data": document_data}
        if chunks:
            chunk_texts = []
            chunk_embeddings = []
            chunk_metadata = []
            for chunk in chunks:
                chunk_texts.append(chunk.get("text", ""))
                chunk_embeddings.append(chunk.get("embedding"))
                chunk_metadata.append(chunk.get("metadata"))
            payload["chunk_texts"] = chunk_texts
            if any(embedding is not None for embedding in chunk_embeddings):
                payload["chunk_embeddings"] = chunk_embeddings
            if any(meta is not None for meta in chunk_metadata):
                payload["chunk_metadata"] = chunk_metadata
        response = self._request(
            "POST", f"/libraries/{library_id}/documents", json=payload
        )
        return Document(**response)

    def get_document(
        self, library_id: Union[str, UUID], document_id: Union[str, UUID]
    ) -> Document:
        """Get document by ID."""
        response = self._request(
            "GET", f"/libraries/{library_id}/documents/{document_id}"
        )
        return Document(**response)

    def list_documents(
        self, library_id: Union[str, UUID], limit: int = 100, offset: int = 0
    ) -> List[Document]:
        """List documents in library."""
        params = {"limit": limit, "offset": offset}
        response = self._request(
            "GET", f"/libraries/{library_id}/documents", params=params
        )
        return [Document(**doc) for doc in response.get("items", [])]

    def delete_document(
        self, library_id: Union[str, UUID], document_id: Union[str, UUID]
    ) -> bool:
        """Delete document."""
        self._request("DELETE", f"/libraries/{library_id}/documents/{document_id}")
        return True

    # Chunk operations

    def create_chunk(
        self,
        library_id: Union[str, UUID],
        document_id: Union[str, UUID],
        text: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        Create a single chunk.

        Args:
            library_id: Library UUID
            document_id: Document UUID
            text: Chunk text
            embedding: Optional embedding vector
            metadata: Optional metadata

        Returns:
            Created chunk object
        """
        data = {"document_id": str(document_id), "text": text}
        if embedding:
            data["embedding"] = embedding
        if metadata:
            data["metadata"] = metadata

        response = self._request("POST", f"/libraries/{library_id}/chunks", json=data)
        return Chunk(**response)

    def get_chunk(
        self, library_id: Union[str, UUID], chunk_id: Union[str, UUID]
    ) -> Chunk:
        """Get chunk by ID."""
        response = self._request("GET", f"/libraries/{library_id}/chunks/{chunk_id}")
        return Chunk(**response)

    def delete_chunk(
        self, library_id: Union[str, UUID], chunk_id: Union[str, UUID]
    ) -> bool:
        """Delete chunk."""
        self._request("DELETE", f"/libraries/{library_id}/chunks/{chunk_id}")
        return True

    # Index operations

    def build_index(
        self,
        library_id: Union[str, UUID],
        index_type: Optional[IndexType] = None,
        force_rebuild: bool = False,
        async_operation: bool = False,
    ) -> Dict[str, Any]:
        """
        Build or rebuild vector index for library.

        Args:
            library_id: Library UUID
            index_type: Type of index to build (auto-selected if None)
            force_rebuild: Force rebuild even if index exists
            async_operation: Run as async operation

        Returns:
            Index operation result

        Example:
            ```python
            # Build HNSW index
            result = client.build_index(
                library_id=library.id,
                index_type=IndexType.HNSW
            )
            print(f"Index built: {result['index_type']}")
            ```
        """
        data = {"force_rebuild": force_rebuild, "async_operation": async_operation}
        if index_type:
            data["index_type"] = index_type.value

        response = self._request("POST", f"/libraries/{library_id}/index", json=data)
        return response

    def get_index_stats(self, library_id: Union[str, UUID]) -> Dict[str, Any]:
        """Get index statistics."""
        response = self._request("GET", f"/libraries/{library_id}/index/stats")
        return response

    # Search operations

    def search(
        self,
        library_id: Union[str, UUID],
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar chunks in library.

        Args:
            library_id: Library UUID
            query_text: Text query (will be converted to embedding)
            query_vector: Pre-computed query vector
            top_k: Number of results to return
            filters: Metadata filters
            min_score: Minimum similarity score threshold

        Returns:
            List of search results

        Example:
            ```python
            # Text search
            results = client.search(
                library_id=library.id,
                query_text="machine learning algorithms",
                top_k=5,
                filters={"category": "research"},
                min_score=0.7
            )

            # Vector search
            results = client.search(
                library_id=library.id,
                query_vector=[0.1, 0.2, 0.3, ...],
                top_k=10
            )
            ```
        """
        if not query_text and not query_vector:
            raise ValueError("Either query_text or query_vector must be provided")

        data = {"top_k": top_k}
        if query_text:
            data["query_text"] = query_text
        if query_vector:
            data["query_vector"] = query_vector
        if filters:
            data["filters"] = filters
        if min_score is not None:
            data["min_score"] = min_score

        response = self._request("POST", f"/libraries/{library_id}/search", json=data)
        results = response.get("results", [])
        return [SearchResult(**r) for r in results]

    # Utility methods

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._request("GET", "/health")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        return self._request("GET", "/statistics")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Any:
        """Make HTTP request to API."""
        url = f"{self.base_url}{path}"

        try:
            response = self.session.request(
                method=method, url=url, params=params, json=json, timeout=self.timeout
            )
            response.raise_for_status()

            # Handle empty responses
            if response.status_code == 204:
                return None

            return response.json()

        except requests.exceptions.HTTPError as e:
            # Extract error message from response
            try:
                error_detail = e.response.json().get("detail", str(e))
            except:
                error_detail = str(e)

            logger.error(f"API error: {error_detail}")
            raise Exception(f"API error: {error_detail}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()