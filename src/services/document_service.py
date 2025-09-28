"""
Document service for business logic.

Handles document management and chunk coordination.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from models.document import Document, DocumentCreate, DocumentUpdate
from models.chunk import Chunk, ChunkMetadata
from repository import DocumentRepository, ChunkRepository
from .embedding_service import EmbeddingService
from .library_service import LibraryService


class DocumentService:
    """
    Service for document operations.

    Manages documents and coordinates chunk creation.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        chunk_repo: ChunkRepository,
        library_service: LibraryService,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize document service.

        Args:
            document_repo: Document repository instance
            chunk_repo: Chunk repository instance
            library_service: Library service instance
            embedding_service: Embedding service instance
        """
        self.document_repo = document_repo
        self.chunk_repo = chunk_repo
        self.library_service = library_service
        self.embedding_service = embedding_service

    async def create_document(
        self,
        library_id: UUID,
        document_data: DocumentCreate,
        chunk_texts: Optional[List[str]] = None,
    ) -> Document:
        """
        Create a document with optional chunks.

        Args:
            library_id: Library to add document to
            document_data: Document creation data
            chunk_texts: Optional list of chunk texts

        Returns:
            Created document
        """
        # Verify library exists
        await self.library_service.get_library(library_id)

        # Create document
        document = Document(
            id=uuid4(),
            library_id=library_id,
            metadata=document_data.metadata,
            chunk_ids=[],
            chunk_count=0,
        )

        # Save document
        document = await self.document_repo.create(document)

        # Create chunks if provided
        if chunk_texts:
            chunks = await self.create_chunks_for_document(
                document.id, chunk_texts, library_id
            )
            document.chunk_ids = [chunk.id for chunk in chunks]
            document.chunk_count = len(chunks)
            await self.document_repo.update(document.id, document)

            # Update library counts
            await self.library_service.update_library_counts(
                library_id, doc_delta=1, chunk_delta=len(chunks)
            )
        else:
            # Just update document count
            await self.library_service.update_library_counts(
                library_id, doc_delta=1
            )

        return document

    async def create_chunks_for_document(
        self, document_id: UUID, chunk_texts: List[str], library_id: Optional[UUID] = None
    ) -> List[Chunk]:
        """
        Create chunks for a document.

        Args:
            document_id: Document ID
            chunk_texts: List of chunk texts
            library_id: Library ID for indexing association

        Returns:
            List of created chunks
        """
        chunks = []

        # Generate embeddings for all texts
        embeddings = await self.embedding_service.generate_embeddings_batch(
            chunk_texts
        )

        # Create chunks
        for position, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk = Chunk(
                id=uuid4(),
                text=text,
                document_id=document_id,
                embedding=embedding,
                metadata=ChunkMetadata(position=position),
            )
            chunk = await self.chunk_repo.create(chunk)

            # Associate chunk with library for indexing
            if library_id:
                await self.chunk_repo.add_to_library_index(library_id, chunk.id)

            chunks.append(chunk)

        return chunks

    async def get_document(self, document_id: UUID) -> Document:
        """
        Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document instance
        """
        return await self.document_repo.get(document_id)

    async def get_documents_by_library(self, library_id: UUID) -> List[Document]:
        """
        Get all documents in a library.

        Args:
            library_id: Library ID

        Returns:
            List of documents
        """
        return await self.document_repo.get_by_library(library_id)

    async def update_document(
        self, document_id: UUID, update_data: DocumentUpdate
    ) -> Document:
        """
        Update document metadata.

        Args:
            document_id: Document ID
            update_data: Update data

        Returns:
            Updated document
        """
        document = await self.document_repo.get(document_id)

        if update_data.metadata is not None:
            document.metadata = update_data.metadata
            document.metadata.updated_at = datetime.utcnow()

        return await self.document_repo.update(document_id, document)

    async def delete_document(self, document_id: UUID) -> bool:
        """
        Delete document and all its chunks.

        Args:
            document_id: Document ID

        Returns:
            True if deleted
        """
        document = await self.document_repo.get(document_id)

        # Delete all chunks
        chunk_count = await self.chunk_repo.delete_by_document(document_id)

        # Delete document
        result = await self.document_repo.delete(document_id)

        # Update library counts
        if result:
            await self.library_service.update_library_counts(
                document.library_id, doc_delta=-1, chunk_delta=-chunk_count
            )

        return result

    async def get_document_with_chunks(self, document_id: UUID) -> dict:
        """
        Get document with all its chunks.

        Args:
            document_id: Document ID

        Returns:
            Document with chunks
        """
        document = await self.document_repo.get(document_id)
        chunks = await self.chunk_repo.get_by_document(document_id)

        return {
            "document": document,
            "chunks": chunks,
        }

    async def list_documents(
        self, library_id: UUID, offset: int = 0, limit: int = 100
    ) -> List[Document]:
        """
        List documents in a library.

        Args:
            library_id: Library ID
            offset: Number of documents to skip
            limit: Maximum documents to return

        Returns:
            List of documents
        """
        # Verify library exists
        await self.library_service.get_library(library_id)

        return await self.document_repo.get_by_library(library_id, offset, limit)

    async def update_document(
        self, document_id: UUID, document_data: DocumentUpdate
    ) -> Document:
        """
        Update document metadata.

        Args:
            document_id: Document ID
            document_data: Update data

        Returns:
            Updated document
        """
        document = await self.document_repo.get(document_id)

        # Update fields if provided
        if document_data.metadata is not None:
            document.metadata = document_data.metadata

        document.updated_at = datetime.now()

        return await self.document_repo.update(document_id, document)

    async def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document ID

        Returns:
            True if deleted successfully
        """
        document = await self.document_repo.get(document_id)

        # Delete all chunks in this document
        await self.chunk_repo.delete_by_document(document_id)

        # Delete the document
        await self.document_repo.delete(document_id)

        # Update library counts
        await self.library_service.update_library_counts(
            document.library_id, doc_delta=-1, chunk_delta=-document.chunk_count
        )

        return True