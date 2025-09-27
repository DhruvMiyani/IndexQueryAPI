"""
Search service for vector similarity queries.

Handles query orchestration, filtering, and result formatting.
"""

from typing import Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime

from models.search import SearchRequest, SearchResult, SearchResponse
from repository import ChunkRepository, DocumentRepository, LibraryRepository
from indexes import BaseIndex
from .embedding_service import EmbeddingService
from .index_service import IndexService


class SearchService:
    """
    Service for vector similarity search.

    Orchestrates search queries across libraries.
    """

    def __init__(
        self,
        chunk_repo: ChunkRepository,
        document_repo: DocumentRepository,
        library_repo: LibraryRepository,
        index_service: IndexService,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize search service.

        Args:
            chunk_repo: Chunk repository instance
            document_repo: Document repository instance
            library_repo: Library repository instance
            index_service: Index service instance
            embedding_service: Embedding service instance
        """
        self.chunk_repo = chunk_repo
        self.document_repo = document_repo
        self.library_repo = library_repo
        self.index_service = index_service
        self.embedding_service = embedding_service

    async def search(
        self,
        library_id: UUID,
        request: SearchRequest,
    ) -> SearchResponse:
        """
        Perform vector similarity search.

        Args:
            library_id: Library to search in
            request: Search request parameters

        Returns:
            Search response with results
        """
        start_time = datetime.utcnow()

        # Get library
        library = await self.library_repo.get(library_id)

        # Get or generate query vector
        if request.query_text:
            query_vector = await self.embedding_service.generate_embedding(
                request.query_text
            )
            query_str = request.query_text
        else:
            query_vector = request.query_vector
            query_str = "vector query"

        # Get index for library
        index = await self.index_service.get_index(library_id)

        if not index:
            # Fall back to linear search if no index
            results = await self._linear_search(
                library_id,
                query_vector,
                request.top_k,
                request.filters,
            )
            index_type = "linear_fallback"
        else:
            # Use index for search
            results = await self._indexed_search(
                index,
                query_vector,
                request.top_k,
                request.filters,
            )
            index_type = library.index_type

        # Apply score threshold if provided
        if request.min_score:
            results = [r for r in results if r.score >= request.min_score]

        # Format results
        formatted_results = await self._format_results(
            results,
            request.include_text,
            request.include_metadata,
        )

        # Calculate search time
        search_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return SearchResponse(
            query=query_str,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=search_time_ms,
            index_type=index_type,
            filters_applied=request.filters,
        )

    async def _linear_search(
        self,
        library_id: UUID,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform linear search without index.

        Args:
            library_id: Library ID
            query_vector: Query vector
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        # Get all chunks in library
        chunks = await self.chunk_repo.get_by_library(library_id)

        # Apply filters if provided
        if filters:
            chunks = self._apply_filters(chunks, filters)

        # Calculate similarities
        results = []
        for chunk in chunks:
            score = self._cosine_similarity(query_vector, chunk.embedding)
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    score=score,
                )
            )

        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def _indexed_search(
        self,
        index: BaseIndex,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform search using index.

        Args:
            index: Index instance
            query_vector: Query vector
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        # Get more results if filtering (to account for filtered out results)
        k_search = top_k * 3 if filters else top_k

        # Search using index
        index_results = index.search(query_vector, k_search)

        # Convert index results to search results and apply filters if provided
        final_results = []
        for result in index_results:
            chunk = await self.chunk_repo.get(result.chunk_id)

            # Apply filters if provided
            if filters and not self._matches_filters(chunk, filters):
                continue

            # Convert to models.SearchResult
            search_result = SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                score=result.score,
            )
            final_results.append(search_result)

            # Stop when we have enough results
            if len(final_results) >= top_k:
                break

        return final_results

    def _apply_filters(self, chunks, filters: Dict):
        """Apply metadata filters to chunks."""
        filtered = []
        for chunk in chunks:
            if self._matches_filters(chunk, filters):
                filtered.append(chunk)
        return filtered

    def _matches_filters(self, chunk, filters: Dict) -> bool:
        """Check if chunk matches all filters."""
        metadata = chunk.metadata.__dict__ if hasattr(chunk.metadata, '__dict__') else {}

        for key, value in filters.items():
            # Handle nested metadata
            if key not in metadata:
                return False

            # Handle different filter types
            if isinstance(value, list):
                # Value must be in list
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range or comparison filters
                if "gte" in value and metadata[key] < value["gte"]:
                    return False
                if "lte" in value and metadata[key] > value["lte"]:
                    return False
                if "eq" in value and metadata[key] != value["eq"]:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False

        return True

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    async def _format_results(
        self,
        results: List[SearchResult],
        include_text: bool,
        include_metadata: bool,
    ) -> List[SearchResult]:
        """
        Format search results with optional data.

        Args:
            results: Raw search results
            include_text: Whether to include chunk text
            include_metadata: Whether to include metadata

        Returns:
            Formatted results
        """
        formatted = []

        for result in results:
            # Get chunk data if needed
            if include_text or include_metadata:
                chunk = await self.chunk_repo.get(result.chunk_id)
                document = await self.document_repo.get(chunk.document_id)

                # Add text if requested
                if include_text:
                    result.text = chunk.text

                # Add metadata if requested
                if include_metadata:
                    result.metadata = chunk.metadata.model_dump()
                    result.document_metadata = document.metadata.model_dump()

            formatted.append(result)

        return formatted