"""
Unit tests for repository implementations.

Tests the in-memory repository implementations for thread safety and correctness.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import UUID, uuid4

# Import our repositories
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from repository import (
    NotFoundError,
    AlreadyExistsError,
    InMemoryLibraryRepository,
    InMemoryDocumentRepository,
    InMemoryChunkRepository,
)
from models.library import Library, LibraryMetadata
from models.document import Document, DocumentMetadata
from models.chunk import Chunk, ChunkMetadata


class TestLibraryRepository:
    """Test LibraryRepository implementation."""

    @pytest.mark.asyncio
    async def test_create_and_get_library(self):
        """Test creating and retrieving a library."""
        repo = InMemoryLibraryRepository()

        # Create library
        library = Library(
            id=uuid4(),
            name="Test Library",
            document_count=0,
            chunk_count=0
        )

        created = await repo.create(library)
        assert created.id == library.id
        assert created.name == library.name

        # Get library
        retrieved = await repo.get(library.id)
        assert retrieved.id == library.id
        assert retrieved.name == library.name

    @pytest.mark.asyncio
    async def test_library_not_found(self):
        """Test getting non-existent library raises error."""
        repo = InMemoryLibraryRepository()
        fake_id = uuid4()

        with pytest.raises(NotFoundError) as exc_info:
            await repo.get(fake_id)

        assert exc_info.value.entity_type == "Library"
        assert exc_info.value.entity_id == fake_id

    @pytest.mark.asyncio
    async def test_duplicate_library_id(self):
        """Test creating library with duplicate ID raises error."""
        repo = InMemoryLibraryRepository()

        library = Library(
            id=uuid4(),
            name="Library 1",
            document_count=0,
            chunk_count=0
        )

        await repo.create(library)

        # Try to create with same ID
        duplicate = Library(
            id=library.id,
            name="Library 2",
            document_count=0,
            chunk_count=0
        )

        with pytest.raises(AlreadyExistsError):
            await repo.create(duplicate)

    @pytest.mark.asyncio
    async def test_unique_library_names(self):
        """Test that library names must be unique."""
        repo = InMemoryLibraryRepository()

        library1 = Library(
            id=uuid4(),
            name="Unique Name",
            document_count=0,
            chunk_count=0
        )
        await repo.create(library1)

        # Try to create with same name
        library2 = Library(
            id=uuid4(),
            name="Unique Name",
            document_count=0,
            chunk_count=0
        )

        with pytest.raises(Exception) as exc_info:
            await repo.create(library2)

        assert "already exists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_library(self):
        """Test updating a library."""
        repo = InMemoryLibraryRepository()

        # Create library
        library = Library(
            id=uuid4(),
            name="Original Name",
            document_count=0,
            chunk_count=0
        )
        await repo.create(library)

        # Update library
        updated = Library(
            id=library.id,
            name="Updated Name",
            document_count=5,
            chunk_count=10,
            metadata=LibraryMetadata(description="Updated description")
        )

        result = await repo.update(library.id, updated)
        assert result.name == "Updated Name"
        assert result.document_count == 5
        assert result.metadata.description == "Updated description"

        # Verify update persisted
        retrieved = await repo.get(library.id)
        assert retrieved.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_library(self):
        """Test deleting a library."""
        repo = InMemoryLibraryRepository()

        library = Library(
            id=uuid4(),
            name="To Delete",
            document_count=0,
            chunk_count=0
        )
        await repo.create(library)

        # Delete library
        result = await repo.delete(library.id)
        assert result is True

        # Verify deleted
        with pytest.raises(NotFoundError):
            await repo.get(library.id)

    @pytest.mark.asyncio
    async def test_list_libraries_with_pagination(self):
        """Test listing libraries with pagination."""
        repo = InMemoryLibraryRepository()

        # Create multiple libraries
        libraries = []
        for i in range(5):
            lib = Library(
                id=uuid4(),
                name=f"Library {i}",
                document_count=0,
                chunk_count=0
            )
            await repo.create(lib)
            libraries.append(lib)

        # List all
        all_libs = await repo.list()
        assert len(all_libs) == 5

        # List with limit
        limited = await repo.list(limit=2)
        assert len(limited) == 2

        # List with offset
        offset_libs = await repo.list(offset=3)
        assert len(offset_libs) == 2

        # List with limit and offset
        paginated = await repo.list(limit=2, offset=2)
        assert len(paginated) == 2

    @pytest.mark.asyncio
    async def test_get_library_by_name(self):
        """Test getting library by name."""
        repo = InMemoryLibraryRepository()

        library = Library(
            id=uuid4(),
            name="Named Library",
            document_count=0,
            chunk_count=0
        )
        await repo.create(library)

        # Get by name
        found = await repo.get_by_name("Named Library")
        assert found is not None
        assert found.id == library.id

        # Get non-existent
        not_found = await repo.get_by_name("Non-existent")
        assert not_found is None


class TestDocumentRepository:
    """Test DocumentRepository implementation."""

    @pytest.mark.asyncio
    async def test_create_and_get_document(self):
        """Test creating and retrieving a document."""
        repo = InMemoryDocumentRepository()

        doc = Document(
            id=uuid4(),
            library_id=uuid4(),
            chunk_ids=[],
            chunk_count=0,
            metadata=DocumentMetadata(title="Test Document")
        )

        created = await repo.create(doc)
        assert created.id == doc.id

        retrieved = await repo.get(doc.id)
        assert retrieved.metadata.title == "Test Document"

    @pytest.mark.asyncio
    async def test_get_documents_by_library(self):
        """Test getting all documents in a library."""
        repo = InMemoryDocumentRepository()
        library_id = uuid4()

        # Create documents in library
        docs = []
        for i in range(3):
            doc = Document(
                id=uuid4(),
                library_id=library_id,
                chunk_ids=[],
                chunk_count=0,
                metadata=DocumentMetadata(title=f"Doc {i}")
            )
            await repo.create(doc)
            docs.append(doc)

        # Create document in different library
        other_doc = Document(
            id=uuid4(),
            library_id=uuid4(),
            chunk_ids=[],
            chunk_count=0,
            metadata=DocumentMetadata(title="Other")
        )
        await repo.create(other_doc)

        # Get documents in library
        lib_docs = await repo.get_by_library(library_id)
        assert len(lib_docs) == 3
        assert all(d.library_id == library_id for d in lib_docs)

    @pytest.mark.asyncio
    async def test_delete_documents_by_library(self):
        """Test deleting all documents in a library."""
        repo = InMemoryDocumentRepository()
        library_id = uuid4()

        # Create documents
        for i in range(3):
            doc = Document(
                id=uuid4(),
                library_id=library_id,
                chunk_ids=[],
                chunk_count=0,
                metadata=DocumentMetadata(title=f"Doc {i}")
            )
            await repo.create(doc)

        # Delete by library
        count = await repo.delete_by_library(library_id)
        assert count == 3

        # Verify deleted
        lib_docs = await repo.get_by_library(library_id)
        assert len(lib_docs) == 0


class TestChunkRepository:
    """Test ChunkRepository implementation."""

    @pytest.mark.asyncio
    async def test_create_and_get_chunk(self):
        """Test creating and retrieving a chunk."""
        repo = InMemoryChunkRepository()

        chunk = Chunk(
            id=uuid4(),
            text="Test content",
            document_id=uuid4(),
            embedding=[0.1, 0.2, 0.3],
            metadata=ChunkMetadata(position=0)
        )

        created = await repo.create(chunk)
        assert created.id == chunk.id

        retrieved = await repo.get(chunk.id)
        assert retrieved.text == "Test content"
        assert retrieved.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_chunks_by_document(self):
        """Test getting chunks by document."""
        repo = InMemoryChunkRepository()
        doc_id = uuid4()

        # Create chunks in document
        chunks = []
        for i in range(3):
            chunk = Chunk(
                id=uuid4(),
                text=f"Chunk {i}",
                document_id=doc_id,
                embedding=[float(i)],
                metadata=ChunkMetadata(position=i)
            )
            await repo.create(chunk)
            chunks.append(chunk)

        # Get by document
        doc_chunks = await repo.get_by_document(doc_id)
        assert len(doc_chunks) == 3
        # Should be sorted by position
        assert doc_chunks[0].metadata.position == 0
        assert doc_chunks[2].metadata.position == 2

    @pytest.mark.asyncio
    async def test_get_vectors_by_library(self):
        """Test getting vectors for library indexing."""
        repo = InMemoryChunkRepository()
        library_id = uuid4()

        # Create chunks and add to library index
        chunk_ids = []
        for i in range(3):
            chunk = Chunk(
                id=uuid4(),
                text=f"Chunk {i}",
                document_id=uuid4(),
                embedding=[float(i), float(i+1)],
                metadata=ChunkMetadata(position=i)
            )
            await repo.create(chunk)
            await repo.add_to_library_index(library_id, chunk.id)
            chunk_ids.append(chunk.id)

        # Get vectors
        vectors = await repo.get_vectors_by_library(library_id)
        assert len(vectors) == 3

        # Check structure
        for chunk_id, embedding in vectors:
            assert isinstance(chunk_id, UUID)
            assert isinstance(embedding, list)
            assert len(embedding) == 2


class TestRepositoryConcurrency:
    """Test repository thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_creates(self):
        """Test concurrent create operations are thread-safe."""
        repo = InMemoryLibraryRepository()

        async def create_library(i: int):
            library = Library(
                id=uuid4(),
                name=f"Library {i}",
                document_count=0,
                chunk_count=0
            )
            await repo.create(library)

        # Create libraries concurrently
        tasks = [create_library(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all created
        count = await repo.count()
        assert count == 10

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test concurrent update operations are thread-safe."""
        repo = InMemoryLibraryRepository()

        # Create library
        library = Library(
            id=uuid4(),
            name="Original",
            document_count=0,
            chunk_count=0
        )
        await repo.create(library)

        async def update_library(count: int):
            updated = Library(
                id=library.id,
                name="Original",
                document_count=count,
                chunk_count=count * 2,
                metadata=library.metadata
            )
            await repo.update(library.id, updated)

        # Update concurrently
        tasks = [update_library(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Final state should be from one of the updates
        final = await repo.get(library.id)
        assert final.document_count >= 0
        assert final.document_count < 10