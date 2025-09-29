"""
Tests for RESTful API best practice: Standardized Pagination.

Tests pagination across all collection endpoints with consistent behavior.
"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from main import app

client = TestClient(app)


class TestPagination:
    """Test standardized pagination implementation."""

    def test_library_pagination_default_params(self):
        """Test library list with default pagination parameters."""
        response = client.get("/libraries")

        assert response.status_code == 200
        data = response.json()

        # Verify pagination structure
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_next" in data
        assert "has_previous" in data

        # Verify default values
        assert data["limit"] == 25
        assert data["offset"] == 0
        assert data["has_previous"] is False
        assert isinstance(data["items"], list)

    def test_library_pagination_custom_params(self):
        """Test library list with custom pagination parameters."""
        response = client.get("/libraries?limit=5&offset=10")

        assert response.status_code == 200
        data = response.json()

        assert data["limit"] == 5
        assert data["offset"] == 10

    def test_library_pagination_invalid_params(self):
        """Test pagination with invalid parameters."""
        # Test limit too high
        response = client.get("/libraries?limit=200")
        assert response.status_code == 422  # Validation error

        # Test negative offset
        response = client.get("/libraries?offset=-1")
        assert response.status_code == 422  # Validation error

        # Test zero limit
        response = client.get("/libraries?limit=0")
        assert response.status_code == 422  # Validation error

    def test_pagination_mathematics(self):
        """Test pagination mathematics are correct."""
        # Create test library first
        library_data = {
            "name": "Test Pagination Library",
            "metadata": {
                "description": "Test library for pagination",
                "tags": ["test"]
            }
        }
        create_response = client.post("/libraries", json=library_data)
        assert create_response.status_code == 201
        library_id = create_response.json()["id"]

        try:
            # Test has_next logic
            response = client.get("/libraries?limit=1&offset=0")
            data = response.json()

            if data["total"] > 1:
                assert data["has_next"] is True
            else:
                assert data["has_next"] is False

            # Test has_previous logic
            if data["total"] > 0:
                response = client.get("/libraries?limit=1&offset=1")
                data = response.json()
                assert data["has_previous"] is True

        finally:
            # Cleanup
            client.delete(f"/libraries/{library_id}")

    def test_document_pagination(self):
        """Test document pagination follows same pattern."""
        # Create test library
        library_data = {
            "name": "Test Doc Pagination",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Test document pagination
            response = client.get(f"/libraries/{library_id}/documents?limit=10&offset=0")
            assert response.status_code == 200

            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert data["limit"] == 10
            assert data["offset"] == 0

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_chunk_pagination(self):
        """Test chunk pagination follows same pattern."""
        # Create test library
        library_data = {
            "name": "Test Chunk Pagination",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Test chunk pagination
            response = client.get(f"/libraries/{library_id}/chunks?limit=15&offset=5")
            assert response.status_code == 200

            data = response.json()
            assert "items" in data
            assert "total" in data
            assert "limit" in data
            assert "offset" in data
            assert data["limit"] == 15
            assert data["offset"] == 5

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_pagination_consistency_across_endpoints(self):
        """Test that all endpoints use consistent pagination structure."""
        # Create test library
        library_data = {
            "name": "Test Consistency",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            endpoints_to_test = [
                "/libraries",
                f"/libraries/{library_id}/documents",
                f"/libraries/{library_id}/chunks"
            ]

            required_fields = {"items", "total", "limit", "offset", "has_next", "has_previous"}

            for endpoint in endpoints_to_test:
                response = client.get(f"{endpoint}?limit=5&offset=0")
                assert response.status_code == 200

                data = response.json()
                actual_fields = set(data.keys())

                # Verify all required pagination fields are present
                assert required_fields.issubset(actual_fields), f"Missing fields in {endpoint}: {required_fields - actual_fields}"

                # Verify field types
                assert isinstance(data["items"], list)
                assert isinstance(data["total"], int)
                assert isinstance(data["limit"], int)
                assert isinstance(data["offset"], int)
                assert isinstance(data["has_next"], bool)
                assert isinstance(data["has_previous"], bool)

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_empty_collection_pagination(self):
        """Test pagination behavior with empty collections."""
        # Create empty library
        library_data = {
            "name": "Empty Library",
            "metadata": {"description": "Empty", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Test empty documents collection
            response = client.get(f"/libraries/{library_id}/documents")
            assert response.status_code == 200

            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0
            assert data["has_next"] is False
            assert data["has_previous"] is False

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_pagination_beyond_total(self):
        """Test pagination when offset exceeds total items."""
        response = client.get("/libraries?limit=10&offset=999999")
        assert response.status_code == 200

        data = response.json()
        assert data["items"] == []
        assert data["has_next"] is False
        # has_previous should be True if offset > 0
        assert data["has_previous"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])