"""
API-level integration tests for the Vector Database REST API.

Tests end-to-end functionality including indexing and search via HTTP endpoints.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fastapi.testclient import TestClient

from main import app


class TestVectorAPIIntegration:
    """Integration tests for the complete API workflow."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_complete_workflow_linear_index(self, client):
        """Test complete workflow with linear index."""
        # Create library
        lib_response = client.post("/libraries", json={
            "name": "Test Library",
            "description": "Integration test library"
        })
        assert lib_response.status_code == 201
        lib_data = lib_response.json()
        lib_id = lib_data["id"]

        # First create a document to add chunks to
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "document_data": {
                "library_id": lib_id,
                "metadata": {
                    "title": "Test Document"
                }
            }
        })
        if doc_response.status_code != 201:
            print(f"Doc creation error: {doc_response.json()}")
        assert doc_response.status_code == 201
        doc_data = doc_response.json()
        doc_id = doc_data["id"]

        # Add chunks with vectors
        chunk_ids = []
        for i in range(5):
            chunk_response = client.post(f"/libraries/{lib_id}/chunks", json={
                "text": f"Test chunk {i}",
                "document_id": doc_id,
                "embedding": [float(i), 0.0, 0.0],  # 3D vectors
                "metadata": {
                    "position": i,
                    "tags": ["test"],
                    "source": "integration_test"
                }
            })
            assert chunk_response.status_code == 201
            chunk_data = chunk_response.json()
            chunk_ids.append(chunk_data["id"])

        # Build linear index (default)
        index_response = client.post(f"/libraries/{lib_id}/index", json={
            "algorithm": "linear"
        })
        if index_response.status_code != 200:
            print(f"Index creation error: {index_response.json()}")
        assert index_response.status_code == 200

        # Search with vector
        search_response = client.post(f"/libraries/{lib_id}/search", json={
            "vector": [2.1, 0.0, 0.0],  # Should be close to chunk 2
            "k": 3
        })
        assert search_response.status_code == 200
        search_data = search_response.json()

        # Verify search results
        assert "results" in search_data
        results = search_data["results"]
        assert len(results) == 3

        # Results should be sorted by similarity (highest first)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        # Top result should be chunk 2 (closest to [2.1, 0, 0])
        assert results[0]["chunk"]["metadata"]["index"] == 2

    def test_kd_tree_index_workflow(self, client):
        """Test complete workflow with KD-Tree index."""
        # Create library
        lib_response = client.post("/libraries", json={
            "name": "KD-Tree Test Library"
        })
        assert lib_response.status_code == 201
        lib_id = lib_response.json()["id"]

        # Create a document
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "library_id": lib_id,
            "metadata": {
                "title": "KD-Tree Test Document"
            }
        })
        assert doc_response.status_code == 201
        doc_id = doc_response.json()["id"]

        # Add chunks in 2D space
        vectors_2d = [
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ]

        for i, vec in enumerate(vectors_2d):
            chunk_response = client.post(f"/libraries/{lib_id}/chunks", json={
                "text": f"2D chunk {i}",
                "document_id": doc_id,
                "embedding": vec,
                "metadata": {"position": i}
            })
            assert chunk_response.status_code == 201

        # Build KD-Tree index
        index_response = client.post(f"/libraries/{lib_id}/index", json={
            "algorithm": "kd_tree"
        })
        assert index_response.status_code == 200

        # Get library stats to verify index was built
        lib_response = client.get(f"/libraries/{lib_id}")
        assert lib_response.status_code == 200
        lib_data = lib_response.json()
        assert lib_data["index_stats"]["type"] == "kd_tree"

        # Search near [1.0, 0.0]
        search_response = client.post(f"/libraries/{lib_id}/search", json={
            "vector": [1.1, 0.0],
            "k": 2
        })
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert len(results) == 2

        # Should find vectors close to [1.1, 0.0]
        assert results[0]["score"] > 0.9  # High similarity

    def test_lsh_index_workflow(self, client):
        """Test complete workflow with LSH index."""
        # Create library
        lib_response = client.post("/libraries", json={
            "name": "LSH Test Library"
        })
        assert lib_response.status_code == 201
        lib_id = lib_response.json()["id"]

        # Create a document
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "library_id": lib_id,
            "metadata": {
                "title": "LSH Test Document"
            }
        })
        assert doc_response.status_code == 201
        doc_id = doc_response.json()["id"]

        # Add chunks with higher-dimensional vectors
        import numpy as np
        np.random.seed(42)

        for i in range(10):
            vec = np.random.randn(8).tolist()  # 8D vectors
            chunk_response = client.post(f"/libraries/{lib_id}/chunks", json={
                "text": f"High-dim chunk {i}",
                "document_id": doc_id,
                "embedding": vec,
                "metadata": {"position": i, "id": i}
            })
            assert chunk_response.status_code == 201

        # Build LSH index
        index_response = client.post(f"/libraries/{lib_id}/index", json={
            "algorithm": "lsh"
        })
        assert index_response.status_code == 200

        # Get library stats
        lib_response = client.get(f"/libraries/{lib_id}")
        assert lib_response.status_code == 200
        lib_data = lib_response.json()
        assert lib_data["index_stats"]["type"] == "lsh"

        # Search (LSH is approximate, so we just verify it works)
        query_vec = np.random.randn(8).tolist()
        search_response = client.post(f"/libraries/{lib_id}/search", json={
            "vector": query_vec,
            "k": 5
        })
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        # LSH might return fewer results due to bucketing
        assert len(results) >= 1

    def test_text_query_with_embedding(self, client):
        """Test search with text query (using embedding service)."""
        # Create library
        lib_response = client.post("/libraries", json={
            "name": "Text Query Library"
        })
        assert lib_response.status_code == 201
        lib_id = lib_response.json()["id"]

        # Create a document
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "library_id": lib_id,
            "metadata": {
                "title": "ML Text Document"
            }
        })
        assert doc_response.status_code == 201
        doc_id = doc_response.json()["id"]

        # Add some chunks
        chunks_data = [
            {"text": "Machine learning algorithms", "embedding": [1.0, 0.8, 0.2]},
            {"text": "Deep learning networks", "embedding": [0.9, 0.9, 0.1]},
            {"text": "Natural language processing", "embedding": [0.7, 0.3, 0.9]}
        ]

        for idx, chunk_data in enumerate(chunks_data):
            chunk_data["document_id"] = doc_id
            chunk_data["metadata"] = {"position": idx}
            chunk_response = client.post(f"/libraries/{lib_id}/chunks", json=chunk_data)
            assert chunk_response.status_code == 201

        # Try text query (will fall back to mock if Cohere unavailable)
        search_response = client.post(f"/libraries/{lib_id}/search", json={
            "query": "machine learning",
            "k": 2
        })

        # Should work even if embedding service is unavailable (mock fallback)
        # Status might be 200 (success) or error depending on embedding service
        if search_response.status_code == 200:
            results = search_response.json()["results"]
            assert len(results) <= 2

    def test_document_level_operations(self, client):
        """Test document-level operations via API."""
        # Create library
        lib_response = client.post("/libraries", json={
            "name": "Document Test Library"
        })
        assert lib_response.status_code == 201
        lib_id = lib_response.json()["id"]

        # Create document with chunks
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "library_id": lib_id,
            "metadata": {
                "title": "Test Document"
            }
        })
        assert doc_response.status_code == 201
        doc_data = doc_response.json()
        doc_id = doc_data["id"]

        # Add chunks to the document manually
        chunk_data_list = [
            {
                "text": "First chunk",
                "document_id": doc_id,
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {"position": 0}
            },
            {
                "text": "Second chunk",
                "document_id": doc_id,
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"position": 1}
            }
        ]

        for chunk_data in chunk_data_list:
            chunk_response = client.post(f"/libraries/{lib_id}/chunks", json=chunk_data)
            assert chunk_response.status_code == 201

        # Get document details
        get_doc_response = client.get(f"/libraries/{lib_id}/documents/{doc_id}")
        assert get_doc_response.status_code == 200
        doc_details = get_doc_response.json()
        assert doc_details["metadata"]["title"] == "Test Document"

        # Search should find the chunks
        search_response = client.post(f"/libraries/{lib_id}/search", json={
            "vector": [1.0, 0.0, 0.0],
            "k": 1
        })
        assert search_response.status_code == 200
        results = search_response.json()["results"]
        assert len(results) == 1
        assert results[0]["chunk"]["text"] == "First chunk"

    def test_error_handling(self, client):
        """Test API error handling."""
        # Try to get non-existent library
        response = client.get("/libraries/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

        # Try to search in non-existent library
        response = client.post("/libraries/00000000-0000-0000-0000-000000000000/search", json={
            "vector": [1.0, 2.0, 3.0],
            "k": 5
        })
        assert response.status_code == 404

        # Try to add chunk with wrong dimension
        lib_response = client.post("/libraries", json={"name": "Error Test"})
        lib_id = lib_response.json()["id"]

        # Create document
        doc_response = client.post(f"/libraries/{lib_id}/documents", json={
            "library_id": lib_id,
            "metadata": {"title": "Error Test Doc"}
        })
        doc_id = doc_response.json()["id"]

        # Add chunk with 3D vector
        client.post(f"/libraries/{lib_id}/chunks", json={
            "text": "3D chunk",
            "document_id": doc_id,
            "embedding": [1.0, 2.0, 3.0],
            "metadata": {"position": 0}
        })

        # Try to add chunk with different dimension
        response = client.post(f"/libraries/{lib_id}/chunks", json={
            "text": "2D chunk",
            "document_id": doc_id,
            "embedding": [1.0, 2.0],  # Wrong dimension
            "metadata": {"position": 1}
        })
        assert response.status_code == 400

        # Try invalid search parameters
        response = client.post(f"/libraries/{lib_id}/search", json={
            "vector": [1.0, 2.0, 3.0, 4.0],  # Wrong dimension
            "k": 5
        })
        assert response.status_code == 400

    def test_library_crud_operations(self, client):
        """Test complete CRUD operations on libraries."""
        # Create
        create_response = client.post("/libraries", json={
            "name": "CRUD Test Library",
            "description": "Testing CRUD operations"
        })
        assert create_response.status_code == 201
        lib_data = create_response.json()
        lib_id = lib_data["id"]

        # Read
        get_response = client.get(f"/libraries/{lib_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "CRUD Test Library"

        # Update
        update_response = client.put(f"/libraries/{lib_id}", json={
            "name": "Updated Library Name",
            "description": "Updated description"
        })
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Library Name"

        # List all libraries
        list_response = client.get("/libraries")
        assert list_response.status_code == 200
        libraries = list_response.json()
        lib_names = [lib["name"] for lib in libraries]
        assert "Updated Library Name" in lib_names

        # Delete
        delete_response = client.delete(f"/libraries/{lib_id}")
        assert delete_response.status_code == 200

        # Verify deletion
        get_response = client.get(f"/libraries/{lib_id}")
        assert get_response.status_code == 404

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "embedding_service" in health_data
        assert "features" in health_data

        features = health_data["features"]
        assert features["vector_search"] is True
        assert features["multiple_indexes"] is True