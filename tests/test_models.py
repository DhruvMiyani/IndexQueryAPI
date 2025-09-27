"""
Unit tests for Pydantic models.

These tests verify model validation, serialization, and business logic.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import ValidationError

# Import our models
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.chunk import Chunk, ChunkCreate, ChunkUpdate, ChunkMetadata
from models.document import Document, DocumentCreate, DocumentUpdate, DocumentMetadata
from models.library import Library, LibraryCreate, LibraryUpdate, LibraryMetadata
from models.search import SearchRequest, SearchResult, SearchResponse


class TestChunkModels:
    """Test chunk-related models."""

    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation with defaults."""
        metadata = ChunkMetadata(position=0)

        assert metadata.position == 0
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
        assert metadata.tags == []
        assert metadata.author is None
        assert metadata.language is None
        assert metadata.source is None

    def test_chunk_metadata_validation(self):
        """Test ChunkMetadata validation rules."""
        # Valid metadata
        metadata = ChunkMetadata(
            position=5,
            tags=["ai", "research"],
            author="John Doe",
            language="en",
            source="paper.pdf"
        )
        assert metadata.position == 5
        assert metadata.tags == ["ai", "research"]

        # Invalid position (negative)
        with pytest.raises(ValidationError) as exc_info:
            ChunkMetadata(position=-1)
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

    def test_chunk_create_valid(self):
        """Test valid ChunkCreate model."""
        doc_id = uuid4()
        chunk_data = ChunkCreate(
            text="This is test content.",
            document_id=doc_id,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(position=0)
        )

        assert chunk_data.text == "This is test content."
        assert chunk_data.document_id == doc_id
        assert chunk_data.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert isinstance(chunk_data.metadata, ChunkMetadata)

    def test_chunk_create_validation_errors(self):
        """Test ChunkCreate validation failures."""
        doc_id = uuid4()

        # Empty text should fail
        with pytest.raises(ValidationError):
            ChunkCreate(text="", document_id=doc_id)

        # Missing text should fail
        with pytest.raises(ValidationError):
            ChunkCreate(document_id=doc_id)

        # Invalid document_id type should fail
        with pytest.raises(ValidationError):
            ChunkCreate(text="Valid text", document_id="not-a-uuid")

    def test_chunk_update_partial(self):
        """Test ChunkUpdate allows partial updates."""
        # Only update text
        update = ChunkUpdate(text="New text content")
        assert update.text == "New text content"
        assert update.metadata is None
        assert update.embedding is None

        # Only update metadata
        new_metadata = ChunkMetadata(position=10, tags=["updated"])
        update = ChunkUpdate(metadata=new_metadata)
        assert update.text is None
        assert update.metadata.position == 10
        assert update.embedding is None

    def test_full_chunk_model(self):
        """Test complete Chunk model."""
        chunk_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            id=chunk_id,
            text="Complete chunk content",
            document_id=doc_id,
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(position=0, tags=["test"])
        )

        assert chunk.id == chunk_id
        assert chunk.document_id == doc_id
        assert chunk.text == "Complete chunk content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata.position == 0


class TestDocumentModels:
    """Test document-related models."""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(title="Test Document")

        assert metadata.title == "Test Document"
        assert metadata.author is None
        assert isinstance(metadata.created_at, datetime)
        assert metadata.tags == []

    def test_document_create_valid(self):
        """Test valid DocumentCreate."""
        lib_id = uuid4()
        metadata = DocumentMetadata(
            title="Research Paper",
            author="Jane Smith",
            category="academic"
        )

        doc = DocumentCreate(library_id=lib_id, metadata=metadata)

        assert doc.library_id == lib_id
        assert doc.metadata.title == "Research Paper"
        assert doc.metadata.author == "Jane Smith"

    def test_document_metadata_validation(self):
        """Test DocumentMetadata validation."""
        # Valid file size
        metadata = DocumentMetadata(title="Test", file_size=1024)
        assert metadata.file_size == 1024

        # Negative file size should fail
        with pytest.raises(ValidationError):
            DocumentMetadata(title="Test", file_size=-1)

    def test_full_document_model(self):
        """Test complete Document model."""
        doc_id = uuid4()
        lib_id = uuid4()
        chunk_ids = [uuid4(), uuid4()]

        doc = Document(
            id=doc_id,
            library_id=lib_id,
            chunk_ids=chunk_ids,
            chunk_count=2,
            metadata=DocumentMetadata(title="Full Document")
        )

        assert doc.id == doc_id
        assert doc.library_id == lib_id
        assert len(doc.chunk_ids) == 2
        assert doc.chunk_count == 2


class TestLibraryModels:
    """Test library-related models."""

    def test_library_create_minimal(self):
        """Test LibraryCreate with minimal data."""
        lib = LibraryCreate(name="Test Library")

        assert lib.name == "Test Library"
        assert isinstance(lib.metadata, LibraryMetadata)
        assert lib.metadata.is_public is False

    def test_library_name_validation(self):
        """Test library name validation."""
        # Valid name
        lib = LibraryCreate(name="Valid Library Name")
        assert lib.name == "Valid Library Name"

        # Empty name should fail
        with pytest.raises(ValidationError):
            LibraryCreate(name="")

        # Too long name should fail (assuming max 255 chars)
        long_name = "x" * 256
        with pytest.raises(ValidationError):
            LibraryCreate(name=long_name)

    def test_library_metadata_validation(self):
        """Test LibraryMetadata validation."""
        # Valid vector dimension
        metadata = LibraryMetadata(vector_dimension=512)
        assert metadata.vector_dimension == 512

        # Zero or negative dimension should fail
        with pytest.raises(ValidationError):
            LibraryMetadata(vector_dimension=0)

        with pytest.raises(ValidationError):
            LibraryMetadata(vector_dimension=-1)

    def test_full_library_model(self):
        """Test complete Library model."""
        lib_id = uuid4()
        doc_ids = [uuid4(), uuid4()]

        lib = Library(
            id=lib_id,
            name="Full Library",
            document_ids=doc_ids,
            document_count=2,
            chunk_count=10,
            index_type="kd_tree",
            index_status="ready"
        )

        assert lib.id == lib_id
        assert lib.document_count == 2
        assert lib.chunk_count == 10
        assert lib.index_type == "kd_tree"
        assert lib.index_status == "ready"


class TestSearchModels:
    """Test search-related models."""

    def test_search_request_with_text(self):
        """Test SearchRequest with text query."""
        request = SearchRequest(
            query_text="machine learning",
            top_k=5,
            filters={"author": "John Doe"},
            min_score=0.7
        )

        assert request.query_text == "machine learning"
        assert request.query_vector is None
        assert request.top_k == 5
        assert request.filters == {"author": "John Doe"}
        assert request.min_score == 0.7

    def test_search_request_with_vector(self):
        """Test SearchRequest with vector query."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        request = SearchRequest(
            query_vector=vector,
            top_k=10
        )

        assert request.query_vector == vector
        assert request.query_text is None
        assert request.top_k == 10

    def test_search_request_validation_errors(self):
        """Test SearchRequest validation failures."""
        # Neither text nor vector provided
        with pytest.raises(ValueError, match="Either query_text or query_vector must be provided"):
            SearchRequest()

        # Both text and vector provided
        with pytest.raises(ValueError, match="Only one of query_text or query_vector should be provided"):
            SearchRequest(
                query_text="test",
                query_vector=[0.1, 0.2]
            )

        # Invalid top_k values
        with pytest.raises(ValidationError):
            SearchRequest(query_text="test", top_k=0)

        with pytest.raises(ValidationError):
            SearchRequest(query_text="test", top_k=1001)

        # Invalid min_score range
        with pytest.raises(ValidationError):
            SearchRequest(query_text="test", min_score=-0.1)

        with pytest.raises(ValidationError):
            SearchRequest(query_text="test", min_score=1.1)

    def test_search_result_model(self):
        """Test SearchResult model."""
        chunk_id = uuid4()
        doc_id = uuid4()

        result = SearchResult(
            chunk_id=chunk_id,
            document_id=doc_id,
            score=0.85,
            text="Relevant text content",
            metadata={"tags": ["ai"]},
            document_metadata={"title": "AI Paper"}
        )

        assert result.chunk_id == chunk_id
        assert result.document_id == doc_id
        assert result.score == 0.85
        assert result.text == "Relevant text content"

    def test_search_result_score_validation(self):
        """Test SearchResult score validation."""
        chunk_id = uuid4()
        doc_id = uuid4()

        # Valid scores
        result = SearchResult(chunk_id=chunk_id, document_id=doc_id, score=0.0)
        assert result.score == 0.0

        result = SearchResult(chunk_id=chunk_id, document_id=doc_id, score=1.0)
        assert result.score == 1.0

        # Invalid scores
        with pytest.raises(ValidationError):
            SearchResult(chunk_id=chunk_id, document_id=doc_id, score=-0.1)

        with pytest.raises(ValidationError):
            SearchResult(chunk_id=chunk_id, document_id=doc_id, score=1.1)

    def test_search_response_model(self):
        """Test SearchResponse model."""
        results = [
            SearchResult(
                chunk_id=uuid4(),
                document_id=uuid4(),
                score=0.9
            )
        ]

        response = SearchResponse(
            query="test query",
            results=results,
            total_results=1,
            search_time_ms=25.5,
            index_type="linear",
            filters_applied={"language": "en"}
        )

        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.search_time_ms == 25.5
        assert response.index_type == "linear"


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_chunk_json_serialization(self):
        """Test chunk model JSON serialization."""
        chunk = Chunk(
            id=uuid4(),
            text="Test content",
            document_id=uuid4(),
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(position=0)
        )

        # Serialize to dict
        chunk_dict = chunk.model_dump()
        assert "id" in chunk_dict
        assert chunk_dict["text"] == "Test content"
        assert isinstance(chunk_dict["embedding"], list)

        # Serialize to JSON
        json_str = chunk.model_dump_json()
        assert isinstance(json_str, str)
        assert "Test content" in json_str

    def test_model_from_dict(self):
        """Test creating models from dictionaries."""
        chunk_data = {
            "id": str(uuid4()),
            "text": "From dict",
            "document_id": str(uuid4()),
            "embedding": [0.1, 0.2],
            "metadata": {
                "position": 0,
                "tags": ["test"],
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }

        chunk = Chunk.model_validate(chunk_data)
        assert chunk.text == "From dict"
        assert len(chunk.embedding) == 2
        assert chunk.metadata.position == 0