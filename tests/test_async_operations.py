"""
Tests for RESTful API best practice: Async Operations with 202 Status.

Tests long-running operations that return 202 Accepted with status tracking.
"""

import pytest
from fastapi.testclient import TestClient
import time
import json

from main import app

client = TestClient(app)


class TestAsyncOperations:
    """Test async operations implementation following RESTful patterns."""

    def test_sync_index_building(self):
        """Test synchronous index building (default behavior)."""
        # Create test library with some data
        library_data = {
            "name": "Sync Index Test Library",
            "metadata": {"description": "Test library", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add some chunks to index
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Test Document"}
                },
                "chunk_texts": [
                    "This is test chunk one for indexing.",
                    "This is test chunk two for indexing."
                ]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Build index synchronously (default)
            index_request = {
                "index_type": "linear",
                "force_rebuild": True,
                "async_operation": False
            }
            response = client.post(f"/libraries/{library_id}/index", json=index_request)

            # Should return 200/201 with IndexResponse for sync operations
            assert response.status_code in [200, 201]

            data = response.json()
            assert "message" in data
            assert "index_type" in data
            assert "library_id" in data
            assert "chunk_count" in data
            assert data["library_id"] == library_id

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_async_index_building_202_response(self):
        """Test async index building returns 202 Accepted."""
        # Create test library
        library_data = {
            "name": "Async Index Test Library",
            "metadata": {"description": "Test async indexing", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add chunks to index
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Async Test Document"}
                },
                "chunk_texts": [
                    "Async test chunk one.",
                    "Async test chunk two.",
                    "Async test chunk three."
                ]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Build index asynchronously
            index_request = {
                "index_type": "kd_tree",
                "force_rebuild": True,
                "async_operation": True
            }
            response = client.post(f"/libraries/{library_id}/index", json=index_request)

            # Should return 202 Accepted
            assert response.status_code == 202

            # Should have Location header pointing to operation status
            assert "location" in response.headers
            location = response.headers["location"]
            assert "/api/operations/" in location

            # Response body should contain operation details
            data = response.json()
            assert "operation_id" in data
            assert "status" in data
            assert "operation_type" in data
            assert "resource_id" in data
            assert "created_at" in data
            assert "links" in data

            assert data["status"] == "pending"
            assert data["operation_type"] == "index_build"
            assert data["resource_id"] == library_id

            return data["operation_id"]  # Return for further testing

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_operation_status_tracking(self):
        """Test operation status endpoint."""
        # Create library and start async operation
        library_data = {
            "name": "Status Tracking Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add data and start async index build
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Status Test Doc"}
                },
                "chunk_texts": ["Status tracking test chunk."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Start async operation
            index_request = {
                "index_type": "linear",
                "async_operation": True
            }
            async_response = client.post(f"/libraries/{library_id}/index", json=index_request)
            assert async_response.status_code == 202

            operation_id = async_response.json()["operation_id"]

            # Check operation status
            status_response = client.get(f"/api/operations/{operation_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            assert "operation_id" in status_data
            assert "status" in status_data
            assert "operation_type" in status_data
            assert "resource_id" in status_data
            assert "created_at" in status_data
            assert "links" in status_data

            assert status_data["operation_id"] == operation_id
            assert status_data["operation_type"] == "index_build"
            assert status_data["resource_id"] == library_id

            # Status should be pending or in_progress initially
            assert status_data["status"] in ["pending", "in_progress", "completed"]

            # Wait a bit and check if operation progresses
            time.sleep(0.5)

            status_response2 = client.get(f"/api/operations/{operation_id}")
            status_data2 = status_response2.json()

            # Should eventually complete
            assert status_data2["status"] in ["in_progress", "completed"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_operation_not_found(self):
        """Test operation status for non-existent operation."""
        fake_operation_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/operations/{fake_operation_id}")

        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
        assert fake_operation_id in error_data["detail"]

    def test_operation_cancellation(self):
        """Test operation cancellation endpoint."""
        # Create library and start async operation
        library_data = {
            "name": "Cancellation Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Start async operation
            index_request = {
                "index_type": "linear",
                "async_operation": True
            }
            async_response = client.post(f"/libraries/{library_id}/index", json=index_request)
            operation_id = async_response.json()["operation_id"]

            # Try to cancel operation
            cancel_response = client.delete(f"/api/operations/{operation_id}")

            # Should return 204 No Content or 409 Conflict if already completed
            assert cancel_response.status_code in [204, 409]

            if cancel_response.status_code == 204:
                # Check that operation is marked as cancelled
                status_response = client.get(f"/api/operations/{operation_id}")
                status_data = status_response.json()
                assert status_data["status"] == "cancelled"

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_operation_links_structure(self):
        """Test that operation responses include proper HATEOAS links."""
        library_data = {
            "name": "Links Test Library",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Start async operation
            index_request = {"async_operation": True}
            async_response = client.post(f"/libraries/{library_id}/index", json=index_request)

            data = async_response.json()
            assert "links" in data

            links = data["links"]
            assert isinstance(links, dict)

            # Should have status and cancel links
            assert "status" in links
            assert "cancel" in links

            # Links should be valid URLs
            assert "/api/operations/" in links["status"]
            assert "/api/operations/" in links["cancel"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_multiple_async_operations(self):
        """Test multiple concurrent async operations."""
        library_data = {
            "name": "Multiple Ops Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add some data
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Multi Ops Doc"}
                },
                "chunk_texts": ["Multi operations test chunk."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Start multiple async operations
            operation_ids = []

            for i in range(3):
                index_request = {
                    "index_type": "linear",
                    "force_rebuild": True,
                    "async_operation": True
                }
                response = client.post(f"/libraries/{library_id}/index", json=index_request)
                assert response.status_code == 202

                operation_id = response.json()["operation_id"]
                operation_ids.append(operation_id)

            # Verify all operations are tracked
            for operation_id in operation_ids:
                status_response = client.get(f"/api/operations/{operation_id}")
                assert status_response.status_code == 200

                status_data = status_response.json()
                assert status_data["operation_type"] == "index_build"
                assert status_data["resource_id"] == library_id

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_operation_result_on_completion(self):
        """Test that completed operations include result data."""
        library_data = {
            "name": "Result Test Library",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add data
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Result Test Doc"}
                },
                "chunk_texts": ["Result test chunk."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Start async operation
            index_request = {
                "index_type": "linear",
                "async_operation": True
            }
            async_response = client.post(f"/libraries/{library_id}/index", json=index_request)
            operation_id = async_response.json()["operation_id"]

            # Wait for completion and check result
            max_attempts = 10
            for attempt in range(max_attempts):
                time.sleep(0.2)
                status_response = client.get(f"/api/operations/{operation_id}")
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    # Should have result data
                    assert "result" in status_data
                    assert status_data["result"] is not None

                    result = status_data["result"]
                    assert "message" in result
                    assert "index_type" in result
                    assert "library_id" in result
                    break
                elif status_data["status"] == "failed":
                    # If failed, should have error message
                    assert "error_message" in status_data
                    assert status_data["error_message"] is not None
                    break
            else:
                pytest.fail("Operation did not complete within expected time")

        finally:
            client.delete(f"/libraries/{library_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])