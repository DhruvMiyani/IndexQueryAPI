"""
Test suite for chunk-library association bug.

This test suite verifies that chunks created through the document endpoint
are properly associated with their library for indexing.
"""

import pytest
import pytest_asyncio
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from models.library import Library, LibraryCreate
from models.document import DocumentCreate, DocumentMetadata
from models.chunk import Chunk, ChunkMetadata
from repository import InMemoryLibraryRepository, InMemoryDocumentRepository, InMemoryChunkRepository
from services import LibraryService, DocumentService, ChunkService, SearchService, EmbeddingService
from indexes import IndexFactory, IndexType


@pytest_asyncio.fixture
async def setup_services():
    """Set up services with in-memory repositories."""
    # Create repositories
    library_repo = InMemoryLibraryRepository()
    document_repo = InMemoryDocumentRepository()
    chunk_repo = InMemoryChunkRepository()

    # Mock embedding service
    embedding_service = AsyncMock(spec=EmbeddingService)
    embedding_service.dimension = 5  # Small dimension for testing
    embedding_service.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    embedding_service.generate_embeddings_batch.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0. 6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]

    # Create services
    library_service = LibraryService(library_repo, document_repo, chunk_repo)
    document_service = DocumentService(
        document_repo,
        chunk_repo,
        library_service,
        embedding_service
    )
    chunk_service = ChunkService(
        chunk_repo,
        embedding_service,
        library_service
    )
    search_service = SearchService(
        library_repo,
        chunk_repo,
        embedding_service,
        IndexFactory()
    )

    return {
        "library_service": library_service,
        "document_service": document_service,
        "chunk_service": chunk_service,
        "search_service": search_service,
        "chunk_repo": chunk_repo,
        "library_repo": library_repo
    }


@pytest.mark.asyncio
async def test_chunks_created_via_document_not_indexed_bug(setup_services):
    """
    Test that demonstrates the bug where chunks created through
    document endpoint are not available for indexing.

    This test should FAIL with the current implementation,
    demonstrating the bug.
    """
    services = await setup_services
    library_service = services["library_service"]
    document_service = services["document_service"]
    search_service = services["search_service"]
    chunk_repo = services["chunk_repo"]

    # Step 1: Create a library
    library = await library_service.create_library(
        LibraryCreate(
            name="Test Library",
            description="Library for testing chunk association",
            index_type=IndexType.LINEAR
        )
    )

    # Step 2: Create a document with chunks
    document = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(
                title="Test Document",
                author="Test Author"
            )
        ),
        chunk_texts=["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
    )

    # Step 3: Verify chunks were created
    assert document.chunk_count == 3
    assert len(document.chunk_ids) == 3

    # Step 4: Try to get chunks by library (This reveals the bug!)
    chunks_in_library = await chunk_repo.get_by_library(library.id)

    # This assertion should FAIL with current implementation
    # because chunks aren't associated with library
    assert len(chunks_in_library) == 3, (
        f"Expected 3 chunks in library, but got {len(chunks_in_library)}. "
        "This indicates chunks created via document endpoint are not "
        "associated with the library!"
    )

    # Step 5: Try to build index (This will also fail!)
    with pytest.raises(ValueError, match="has no chunks to index"):
        await search_service.build_index(library.id, IndexType.LINEAR)


@pytest.mark.asyncio
async def test_chunks_created_directly_are_indexed_correctly(setup_services):
    """
    Test that chunks created directly through chunk endpoint
    ARE properly indexed (contrast with document endpoint).

    This test should PASS, showing the chunk endpoint works correctly.
    """
    services = await setup_services
    library_service = services["library_service"]
    document_service = services["document_service"]
    chunk_service = services["chunk_service"]
    search_service = services["search_service"]
    chunk_repo = services["chunk_repo"]

    # Step 1: Create a library
    library = await library_service.create_library(
        LibraryCreate(
            name="Test Library",
            description="Library for testing direct chunk creation",
            index_type=IndexType.LINEAR
        )
    )

    # Step 2: Create a document (without chunks)
    document = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(
                title="Test Document",
                author="Test Author"
            )
        ),
        chunk_texts=None  # No chunks initially
    )

    # Step 3: Create chunks directly through chunk service
    from models.chunk import ChunkCreate

    for i, text in enumerate(["Chunk 1", "Chunk 2", "Chunk 3"]):
        await chunk_service.create_chunk(
            document_id=document.id,
            chunk_data=ChunkCreate(
                text=text,
                document_id=document.id,
                metadata=ChunkMetadata(position=i)
            ),
            library_id=library.id  # This properly associates with library
        )

    # Step 4: Verify chunks are properly associated with library
    chunks_in_library = await chunk_repo.get_by_library(library.id)
    assert len(chunks_in_library) == 3, "Chunks created directly should be in library"

    # Step 5: Build index should work
    await search_service.build_index(library.id, IndexType.LINEAR)

    # Step 6: Verify search works
    results = await search_service.search(
        library_id=library.id,
        query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=2
    )
    assert len(results) == 2, "Search should return results"


@pytest.mark.asyncio
async def test_chunk_library_association_after_fix(setup_services):
    """
    Test that verifies the fix properly associates chunks with library
    when created through document endpoint.

    This test should PASS after implementing the fix.
    """
    services = await setup_services
    library_service = services["library_service"]
    document_service = services["document_service"]
    search_service = services["search_service"]
    chunk_repo = services["chunk_repo"]

    # Step 1: Create a library
    library = await library_service.create_library(
        LibraryCreate(
            name="Test Library Fixed",
            description="Library for testing the fix",
            index_type=IndexType.LINEAR
        )
    )

    # Step 2: Create document with chunks
    document = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(
                title="Test Document with Fix",
                author="Test Author"
            )
        ),
        chunk_texts=["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    )

    # Step 3: Chunks should be associated with library
    chunks_in_library = await chunk_repo.get_by_library(library.id)
    assert len(chunks_in_library) == 5, (
        "After fix, chunks created via document should be in library"
    )

    # Step 4: Building index should work
    await search_service.build_index(library.id, IndexType.LINEAR)

    # Step 5: Search should work
    results = await search_service.search(
        library_id=library.id,
        query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=3
    )
    assert len(results) == 3, "Search should return top 3 results"

    # Step 6: Verify library stats are correct
    updated_library = await library_service.get_library(library.id)
    assert updated_library.chunk_count == 5
    assert updated_library.document_count == 1


@pytest.mark.asyncio
async def test_mixed_chunk_creation_methods(setup_services):
    """
    Test that both document-created and directly-created chunks
    work together after the fix.
    """
    services = await setup_services
    library_service = services["library_service"]
    document_service = services["document_service"]
    chunk_service = services["chunk_service"]
    search_service = services["search_service"]
    chunk_repo = services["chunk_repo"]

    # Create library
    library = await library_service.create_library(
        LibraryCreate(
            name="Mixed Test Library",
            description="Testing mixed chunk creation",
            index_type=IndexType.LINEAR
        )
    )

    # Create document with chunks
    doc1 = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(title="Document 1")
        ),
        chunk_texts=["Doc1 Chunk1", "Doc1 Chunk2"]
    )

    # Create another document without chunks
    doc2 = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(title="Document 2")
        ),
        chunk_texts=None
    )

    # Add chunks directly to doc2
    from models.chunk import ChunkCreate

    await chunk_service.create_chunk(
        document_id=doc2.id,
        chunk_data=ChunkCreate(
            text="Doc2 Direct Chunk",
            document_id=doc2.id,
            metadata=ChunkMetadata(position=0)
        ),
        library_id=library.id
    )

    # All chunks should be in library
    chunks_in_library = await chunk_repo.get_by_library(library.id)
    assert len(chunks_in_library) == 3, (
        "Should have 2 chunks from doc1 and 1 from doc2"
    )

    # Indexing should work
    await search_service.build_index(library.id, IndexType.LINEAR)

    # Search should find all chunks
    results = await search_service.search(
        library_id=library.id,
        query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        top_k=10
    )
    assert len(results) == 3, "All 3 chunks should be searchable"


@pytest.mark.asyncio
async def test_get_vectors_by_library_after_fix(setup_services):
    """
    Test that get_vectors_by_library works correctly after fix.
    This is crucial for index building.
    """
    services = await setup_services
    library_service = services["library_service"]
    document_service = services["document_service"]
    chunk_repo = services["chunk_repo"]

    # Create library
    library = await library_service.create_library(
        LibraryCreate(
            name="Vector Test Library",
            description="Testing vector retrieval",
            index_type=IndexType.LINEAR
        )
    )

    # Create document with chunks
    document = await document_service.create_document(
        library_id=library.id,
        document_data=DocumentCreate(
            metadata=DocumentMetadata(title="Vector Test Doc")
        ),
        chunk_texts=["Chunk A", "Chunk B", "Chunk C"]
    )

    # Get vectors by library (this is what indexing uses)
    vectors = await chunk_repo.get_vectors_by_library(library.id)

    # Should have 3 vectors
    assert len(vectors) == 3, (
        f"Expected 3 vectors, got {len(vectors)}. "
        "This is what causes 'no chunks to index' error!"
    )

    # Each vector should be (chunk_id, embedding)
    for chunk_id, embedding in vectors:
        assert isinstance(chunk_id, type(uuid4()))
        assert isinstance(embedding, list)
        assert len(embedding) == 5  # Our test dimension