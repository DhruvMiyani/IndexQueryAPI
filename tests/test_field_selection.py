"""
Tests for RESTful API best practice: Field Selection/Projection.

Tests the ability to reduce payload sizes by selecting specific fields.
"""

import pytest
from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)


class TestFieldSelection:
    """Test field selection/projection implementation."""

    def test_library_field_selection_basic(self):
        """Test basic field selection on library endpoints."""
        # Create test library
        library_data = {
            "name": "Field Selection Test Library",
            "metadata": {
                "description": "Test library for field selection",
                "tags": ["test", "field-selection"]
            }
        }
        create_response = client.post("/libraries", json=library_data)
        assert create_response.status_code == 201
        library_id = create_response.json()["id"]

        try:
            # Test selecting specific fields
            response = client.get(f"/libraries?fields=id,name")
            assert response.status_code == 200

            data = response.json()
            assert "items" in data

            if data["items"]:
                item = data["items"][0]
                # Should only have the requested fields (plus any required ones)
                expected_fields = {"id", "name"}
                actual_fields = set(item.keys())

                # At minimum, requested fields should be present
                assert expected_fields.issubset(actual_fields)

                # Should not have metadata field
                assert "metadata" not in item or item["metadata"] is None

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_library_field_selection_all_fields(self):
        """Test that without field selection, all fields are returned."""
        library_data = {
            "name": "Complete Library Test",
            "metadata": {
                "description": "Complete library",
                "tags": ["complete"]
            }
        }
        create_response = client.post("/libraries", json=library_data)
        library_id = create_response.json()["id"]

        try:
            # Without field selection - should get all fields
            response = client.get("/libraries")
            assert response.status_code == 200

            data = response.json()
            if data["items"]:
                item = data["items"][0]

                # Should have core library fields
                expected_fields = {"id", "name", "metadata", "document_count", "chunk_count"}
                actual_fields = set(item.keys())
                assert expected_fields.issubset(actual_fields)

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_chunk_field_selection_exclude_embeddings(self):
        """Test field selection on chunks to exclude large embedding vectors."""
        # Create test library and document with chunks
        library_data = {
            "name": "Chunk Field Test Library",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Create document with chunks
            doc_data = {
                "document_data": {
                    "metadata": {
                        "title": "Test Document",
                        "author": "Test Author"
                    }
                },
                "chunk_texts": ["This is test chunk text for field selection testing."]
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=doc_data)
            assert doc_response.status_code == 201

            # Test field selection excluding embeddings
            response = client.get(f"/libraries/{library_id}/chunks?fields=id,text,metadata")
            assert response.status_code == 200

            data = response.json()
            if data["items"]:
                chunk = data["items"][0]

                # Should have requested fields
                expected_fields = {"id", "text", "metadata"}
                actual_fields = set(chunk.keys())
                assert expected_fields.issubset(actual_fields)

                # Should NOT have embedding field (saves bandwidth)
                assert "embedding" not in chunk

            # Test that without field selection, embedding is included
            response_all = client.get(f"/libraries/{library_id}/chunks")
            assert response_all.status_code == 200

            data_all = response_all.json()
            if data_all["items"]:
                chunk_all = data_all["items"][0]
                assert "embedding" in chunk_all

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_field_selection_invalid_fields(self):
        """Test field selection with invalid field names."""
        library_data = {
            "name": "Invalid Field Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Test with invalid field names
            response = client.get(f"/libraries?fields=id,invalid_field,another_invalid")

            # Should return 400 Bad Request for invalid fields
            assert response.status_code == 400

            error_data = response.json()
            assert "detail" in error_data
            assert "Invalid fields" in error_data["detail"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_field_selection_empty_fields(self):
        """Test field selection with empty fields parameter."""
        response = client.get("/libraries?fields=")
        assert response.status_code == 200

        # Empty fields should return all fields (same as no fields param)
        data = response.json()
        assert "items" in data

    def test_field_selection_single_item_endpoints(self):
        """Test field selection on single item endpoints."""
        # Create test library
        library_data = {
            "name": "Single Item Field Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Create a chunk to test single item field selection
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Test Doc"}
                },
                "chunk_texts": ["Test chunk for single item field selection."]
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=doc_data)
            doc_id = doc_response.json()["id"]

            # Get chunks to find chunk ID
            chunks_response = client.get(f"/libraries/{library_id}/chunks")
            chunks = chunks_response.json()["items"]
            if chunks:
                chunk_id = chunks[0]["id"]

                # Test field selection on single chunk
                response = client.get(f"/libraries/{library_id}/chunks/{chunk_id}?fields=id,text")
                assert response.status_code == 200

                chunk_data = response.json()
                expected_fields = {"id", "text"}
                actual_fields = set(chunk_data.keys())
                assert expected_fields.issubset(actual_fields)
                assert "embedding" not in chunk_data

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_field_selection_bandwidth_savings(self):
        """Test that field selection actually reduces response size."""
        # Create library with document and chunks
        library_data = {
            "name": "Bandwidth Test Library",
            "metadata": {"description": "Test bandwidth savings", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Create document with multiple chunks
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Bandwidth Test Doc"}
                },
                "chunk_texts": [
                    "This is the first test chunk for bandwidth testing.",
                    "This is the second test chunk for bandwidth testing.",
                    "This is the third test chunk for bandwidth testing."
                ]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Get full response (with embeddings)
            full_response = client.get(f"/libraries/{library_id}/chunks")
            full_data = full_response.json()
            full_size = len(json.dumps(full_data))

            # Get filtered response (without embeddings)
            filtered_response = client.get(f"/libraries/{library_id}/chunks?fields=id,text,metadata")
            filtered_data = filtered_response.json()
            filtered_size = len(json.dumps(filtered_data))

            # Filtered response should be significantly smaller
            assert filtered_size < full_size
            savings_ratio = (full_size - filtered_size) / full_size

            # Should save at least 30% bandwidth (embeddings are large)
            assert savings_ratio > 0.3, f"Expected >30% savings, got {savings_ratio:.2%}"

            print(f"Bandwidth savings: {savings_ratio:.1%} ({full_size} -> {filtered_size} bytes)")

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_field_selection_with_pagination(self):
        """Test field selection works correctly with pagination."""
        library_data = {
            "name": "Pagination + Fields Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Test field selection with pagination
            response = client.get(f"/libraries?fields=id,name&limit=5&offset=0")
            assert response.status_code == 200

            data = response.json()
            # Pagination fields should still be present
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert "has_next" in data
            assert "has_previous" in data

            # Items should only have selected fields
            if data["items"]:
                item = data["items"][0]
                expected_fields = {"id", "name"}
                actual_fields = set(item.keys())
                assert expected_fields.issubset(actual_fields)

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_field_selection_comma_separated_parsing(self):
        """Test various formats of comma-separated field lists."""
        library_data = {
            "name": "CSV Parsing Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            test_cases = [
                "id,name",  # Basic
                "id, name",  # With spaces
                " id , name ",  # Extra spaces
                "id,name,document_count",  # Multiple fields
            ]

            for fields_param in test_cases:
                response = client.get(f"/libraries?fields={fields_param}")
                assert response.status_code == 200

                data = response.json()
                if data["items"]:
                    item = data["items"][0]
                    # Should have at least id and name
                    assert "id" in item
                    assert "name" in item

        finally:
            client.delete(f"/libraries/{library_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])