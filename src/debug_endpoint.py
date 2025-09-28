"""
Temporary debug endpoint to inspect repository state.
Add this to your main.py temporarily to debug the chunk issue.
"""

from fastapi import FastAPI, Depends
from uuid import UUID
from api.dependencies import get_chunk_service, get_library_service
from services import ChunkService, LibraryService

def add_debug_routes(app: FastAPI):
    """Add debug routes to the app."""

    @app.get("/debug/library/{library_id}/chunks")
    async def debug_library_chunks(
        library_id: UUID,
        chunk_service: ChunkService = Depends(get_chunk_service)
    ):
        """Debug: Check what chunks are associated with this library."""
        # Direct repository access
        chunks_in_library = await chunk_service.chunk_repo.get_by_library(library_id)
        all_chunks = await chunk_service.chunk_repo.list()

        # Check the internal mappings
        library_chunks_mapping = getattr(chunk_service.chunk_repo, '_library_chunks', {})

        return {
            "library_id": str(library_id),
            "chunks_in_library": len(chunks_in_library),
            "total_chunks_in_system": len(all_chunks),
            "library_chunks_mapping": {
                str(k): len(v) for k, v in library_chunks_mapping.items()
            },
            "chunks_data": [
                {
                    "id": str(chunk.id),
                    "text": chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text,
                    "document_id": str(chunk.document_id)
                }
                for chunk in chunks_in_library
            ],
            "all_chunks_sample": [
                {
                    "id": str(chunk.id),
                    "text": chunk.text[:30] + "..." if len(chunk.text) > 30 else chunk.text,
                    "document_id": str(chunk.document_id)
                }
                for chunk in all_chunks[:5]  # Just first 5
            ]
        }

    @app.post("/debug/library/{library_id}/fix-chunks")
    async def debug_fix_library_chunks(
        library_id: UUID,
        chunk_service: ChunkService = Depends(get_chunk_service),
        library_service: LibraryService = Depends(get_library_service)
    ):
        """Debug: Manually fix chunks that aren't associated with library."""
        # Get all documents in this library
        documents = await library_service.get_documents_by_library(library_id)

        fixed_count = 0
        for doc in documents:
            # Get chunks for this document
            doc_chunks = await chunk_service.chunk_repo.get_by_document(doc.id)

            for chunk in doc_chunks:
                # Add to library index if not already there
                await chunk_service.chunk_repo.add_to_library_index(library_id, chunk.id)
                fixed_count += 1

        return {
            "library_id": str(library_id),
            "documents_processed": len(documents),
            "chunks_fixed": fixed_count,
            "message": f"Fixed {fixed_count} chunks that weren't associated with library"
        }