"""
Unit tests for service layer.

Tests business logic and orchestration in services.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

# Import test dependencies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services import (
    LibraryService,
    DocumentService,
    ChunkService,
    EmbeddingService,
    IndexService,
)
from models.library import Library, LibraryCreate
from models.document import Document, DocumentCreate, DocumentMetadata
from models.chunk import Chunk, ChunkCreate, ChunkMetadata
from repository import (
    InMemoryLibraryRepository,
    InMemoryDocumentRepository,
    InMemoryChunkRepository,
)


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest.mark.asyncio
    async def test_generate_mock_embedding(self):
        """Test mock embedding generation."""
        service = EmbeddingService(provider="local", dimension=128)

        text = "This is a test"
        embedding = await service.generate_embedding(text)

        assert len(embedding) == 128
        assert isinstance(embedding[0], float)

        # Same text should generate same embedding (deterministic)
        embedding2 = await service.generate_embedding(text)
        assert embedding == embedding2

    @pytest.mark.asyncio
    async def test_embedding_caching(self):
        """Test embedding caching functionality."""
        service = EmbeddingService(provider="local", dimension=64)

        text = "Cache test"

        # First call
        embedding1 = await service.generate_embedding(text)

        # Second call should use cache (same result)
        embedding2 = await service.generate_embedding(text)
        assert embedding1 == embedding2

        # Verify cache contains the text
        assert text in service._cache

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        service = EmbeddingService(provider="local", dimension=32)

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await service.generate_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 32 for emb in embeddings)
        assert embeddings[0] != embeddings[1]  # Should be different

    def test_embedding_service_config(self):
        """Test embedding service configuration."""
        service = EmbeddingService(
            provider="cohere",
            api_key="test-key",
            model="test-model",
            dimension=256
        )

        assert service.provider == "cohere"
        assert service.api_key == "test-key"
        assert service.model == "test-model"
        assert service.dimension == 256


class TestLibraryService:
    """Test LibraryService functionality."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create LibraryService with mock repositories."""
        lib_repo = InMemoryLibraryRepository()
        doc_repo = InMemoryDocumentRepository()
        chunk_repo = InMemoryChunkRepository()
        return LibraryService(lib_repo, doc_repo, chunk_repo)

    @pytest.mark.asyncio
    async def test_create_library(self, service):
        """Test library creation."""
        library_data = LibraryCreate(name="Test Library")

        library = await service.create_library(library_data)

        assert library.name == "Test Library"
        assert library.document_count == 0
        assert library.chunk_count == 0
        assert library.index_status == "none"

    @pytest.mark.asyncio
    async def test_get_library(self, service):
        """Test getting a library."""
        # Create library first
        library_data = LibraryCreate(name="Get Test")
        created = await service.create_library(library_data)

        # Get library
        retrieved = await service.get_library(created.id)

        assert retrieved.id == created.id
        assert retrieved.name == "Get Test"

    @pytest.mark.asyncio
    async def test_list_libraries(self, service):
        """Test listing libraries with pagination."""
        # Create multiple libraries
        for i in range(5):
            library_data = LibraryCreate(name=f"Library {i}")
            await service.create_library(library_data)

        # List all
        all_libs = await service.list_libraries()
        assert len(all_libs) == 5

        # List with limit
        limited = await service.list_libraries(limit=2)
        assert len(limited) == 2

        # List with offset
        offset_libs = await service.list_libraries(offset=3)
        assert len(offset_libs) == 2

    @pytest.mark.asyncio
    async def test_delete_library_cascade(self, service):
        """Test cascade delete of library with documents and chunks."""
        # Create library
        library_data = LibraryCreate(name="Delete Test")
        library = await service.create_library(library_data)

        # Add documents and chunks (simulate)
        doc = Document(
            id=uuid4(),
            library_id=library.id,
            chunk_ids=[],
            chunk_count=0,
            metadata=DocumentMetadata(title="Test Doc")
        )
        await service.document_repo.create(doc)

        chunk = Chunk(
            id=uuid4(),
            text="Test chunk",
            document_id=doc.id,
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(position=0)
        )
        await service.chunk_repo.create(chunk)

        # Delete library (should cascade)
        result = await service.delete_library(library.id)
        assert result is True

        # Verify all deleted
        from repository.base import NotFoundError
        with pytest.raises(NotFoundError):
            await service.get_library(library.id)

    @pytest.mark.asyncio
    async def test_update_library_counts(self, service):
        """Test updating library document/chunk counts."""
        # Create library
        library_data = LibraryCreate(name="Count Test")
        library = await service.create_library(library_data)

        # Update counts
        await service.update_library_counts(
            library.id, doc_delta=2, chunk_delta=10
        )

        # Verify counts updated
        updated = await service.get_library(library.id)
        assert updated.document_count == 2
        assert updated.chunk_count == 10

    @pytest.mark.asyncio
    async def test_get_library_statistics(self, service):
        """Test getting library statistics."""
        # Create library with some data
        library_data = LibraryCreate(name="Stats Test")
        library = await service.create_library(library_data)

        # Add documents
        for i in range(3):
            doc = Document(
                id=uuid4(),
                library_id=library.id,
                chunk_ids=[],
                chunk_count=2,  # Each doc has 2 chunks
                metadata=DocumentMetadata(title=f"Doc {i}")
            )
            await service.document_repo.create(doc)

        # Update library counts
        await service.update_library_counts(
            library.id, doc_delta=3, chunk_delta=6
        )

        # Get statistics
        stats = await service.get_library_statistics(library.id)

        assert stats["document_count"] == 3
        assert stats["chunk_count"] == 6
        assert stats["avg_chunks_per_document"] == 2.0
        assert stats["name"] == "Stats Test"


class TestDocumentService:
    """Test DocumentService functionality."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create DocumentService with dependencies."""
        doc_repo = InMemoryDocumentRepository()
        chunk_repo = InMemoryChunkRepository()

        # Mock library service
        library_service = AsyncMock()
        library_service.get_library.return_value = Library(
            id=uuid4(),
            name="Test Library",
            document_count=0,
            chunk_count=0
        )

        # Mock embedding service
        embedding_service = AsyncMock()
        embedding_service.generate_embeddings_batch.return_value = [
            [0.1, 0.2, 0.3],  # Mock embeddings
            [0.4, 0.5, 0.6],
        ]

        return DocumentService(
            doc_repo, chunk_repo, library_service, embedding_service
        )

    @pytest.mark.asyncio
    async def test_create_document_without_chunks(self, service):
        """Test creating document without chunks."""
        library_id = uuid4()
        doc_data = DocumentCreate(
            library_id=library_id,
            metadata=DocumentMetadata(title="Test Document")
        )

        document = await service.create_document(library_id, doc_data)

        assert document.metadata.title == "Test Document"
        assert document.chunk_count == 0
        assert len(document.chunk_ids) == 0

    @pytest.mark.asyncio
    async def test_create_document_with_chunks(self, service):
        """Test creating document with chunks."""
        library_id = uuid4()
        doc_data = DocumentCreate(
            library_id=library_id,
            metadata=DocumentMetadata(title="Doc with Chunks")
        )
        chunk_texts = ["Chunk 1 text", "Chunk 2 text"]

        document = await service.create_document(
            library_id, doc_data, chunk_texts
        )

        assert document.chunk_count == 2
        assert len(document.chunk_ids) == 2

        # Verify embedding service was called
        service.embedding_service.generate_embeddings_batch.assert_called_once_with(
            chunk_texts
        )

    @pytest.mark.asyncio
    async def test_get_document_with_chunks(self, service):
        """Test getting document with its chunks."""
        # Create document
        library_id = uuid4()
        doc_data = DocumentCreate(
            library_id=library_id,
            metadata=DocumentMetadata(title="Full Document")
        )
        document = await service.create_document(library_id, doc_data)

        # Add a chunk manually
        chunk = Chunk(
            id=uuid4(),
            text="Test chunk",
            document_id=document.id,
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(position=0)
        )
        await service.chunk_repo.create(chunk)

        # Get document with chunks
        result = await service.get_document_with_chunks(document.id)

        assert "document" in result
        assert "chunks" in result
        assert result["document"].id == document.id
        assert len(result["chunks"]) == 1


class TestChunkService:
    """Test ChunkService functionality."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create ChunkService with dependencies."""
        chunk_repo = InMemoryChunkRepository()

        # Mock embedding service
        embedding_service = AsyncMock()
        embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4]
        embedding_service.generate_embeddings_batch.return_value = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]

        # Mock library service
        library_service = AsyncMock()

        return ChunkService(chunk_repo, embedding_service, library_service)

    @pytest.mark.asyncio
    async def test_create_chunk_with_embedding_generation(self, service):
        """Test creating chunk with automatic embedding generation."""
        document_id = uuid4()
        chunk_data = ChunkCreate(
            text="Test chunk text",
            document_id=document_id,
            metadata=ChunkMetadata(position=0)
        )

        chunk = await service.create_chunk(document_id, chunk_data)

        assert chunk.text == "Test chunk text"
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]

        # Verify embedding was generated
        service.embedding_service.generate_embedding.assert_called_once_with(
            "Test chunk text"
        )

    @pytest.mark.asyncio
    async def test_create_chunk_with_provided_embedding(self, service):
        """Test creating chunk with pre-computed embedding."""
        document_id = uuid4()
        chunk_data = ChunkCreate(
            text="Test chunk text",
            document_id=document_id,
            embedding=[1.0, 2.0, 3.0, 4.0],
            metadata=ChunkMetadata(position=0)
        )

        chunk = await service.create_chunk(document_id, chunk_data)

        assert chunk.embedding == [1.0, 2.0, 3.0, 4.0]

        # Verify embedding service was NOT called
        service.embedding_service.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_create_chunks(self, service):
        """Test bulk chunk creation."""
        document_id = uuid4()
        texts = ["Chunk 1", "Chunk 2", "Chunk 3"]

        chunks = await service.bulk_create_chunks(document_id, texts)

        assert len(chunks) == 3
        assert chunks[0].metadata.position == 0
        assert chunks[1].metadata.position == 1
        assert chunks[2].metadata.position == 2

        # Verify batch embedding was called
        service.embedding_service.generate_embeddings_batch.assert_called_once_with(
            texts
        )

    @pytest.mark.asyncio
    async def test_update_chunk_with_text_change(self, service):
        """Test updating chunk text regenerates embedding."""
        # Create chunk first
        document_id = uuid4()
        chunk_data = ChunkCreate(
            text="Original text",
            document_id=document_id,
            metadata=ChunkMetadata(position=0)
        )
        chunk = await service.create_chunk(document_id, chunk_data)

        # Reset mock call count
        service.embedding_service.generate_embedding.reset_mock()

        # Update chunk text
        from models.chunk import ChunkUpdate
        update_data = ChunkUpdate(text="Updated text")

        updated = await service.update_chunk(chunk.id, update_data)

        assert updated.text == "Updated text"

        # Verify embedding was regenerated
        service.embedding_service.generate_embedding.assert_called_once_with(
            "Updated text"
        )


class TestIndexService:
    """Test IndexService functionality."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create IndexService with mock repositories."""
        lib_repo = InMemoryLibraryRepository()
        chunk_repo = InMemoryChunkRepository()
        return IndexService(lib_repo, chunk_repo)

    @pytest.mark.asyncio
    async def test_index_service_initialization(self, service):
        """Test IndexService initializes correctly."""
        assert service.library_repo is not None
        assert service.chunk_repo is not None
        assert len(service._indexes) == 0

    @pytest.mark.asyncio
    async def test_get_index_stats_no_index(self, service):
        """Test getting stats when no index exists."""
        # Create library
        library = Library(
            id=uuid4(),
            name="Test Library",
            document_count=0,
            chunk_count=0,
            index_status="none"
        )
        await service.library_repo.create(library)

        stats = await service.get_index_stats(library.id)

        assert stats["library_id"] == library.id
        assert stats["index_status"] == "none"
        assert stats["index_type"] is None

    @pytest.mark.asyncio
    async def test_clear_all_indexes(self, service):
        """Test clearing all indexes."""
        # Add some mock indexes
        service._indexes[uuid4()] = MagicMock()
        service._indexes[uuid4()] = MagicMock()

        assert len(service._indexes) == 2

        service.clear_all_indexes()

        assert len(service._indexes) == 0