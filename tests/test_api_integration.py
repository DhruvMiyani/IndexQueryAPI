"""
Integration tests for RESTful API best practices.

Tests all API best practices working together in realistic scenarios.
"""

import pytest
from fastapi.testclient import TestClient
import time
import json

from main import app

client = TestClient(app)


class TestAPIIntegration:
    """Integration tests for all RESTful API best practices."""

    def test_complete_workflow_with_all_api_features(self):
        """Test complete workflow using all API best practices together."""
        # Create library
        library_data = {
            "name": "Complete API Features Test",
            "metadata": {"description": "Testing all API features", "tags": ["api", "integration"]}
        }
        lib_response = client.post("/libraries", json=library_data)
        assert lib_response.status_code == 201
        library_id = lib_response.json()["id"]

        try:
            # Test 1: Create documents and verify HATEOAS links
            doc_data = {
                "document_data": {
                    "metadata": {"title": "API Integration Test Doc", "author": "Integration Tester"}
                },
                "chunk_texts": [
                    "This is a test chunk for API integration testing.",
                    "Another test chunk with different content for testing.",
                    "Final test chunk to ensure proper pagination and field selection."
                ]
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=doc_data)
            assert doc_response.status_code == 201

            # Test 2: Test pagination with HATEOAS links
            docs_response = client.get(f"/libraries/{library_id}/documents?limit=2&offset=0")
            assert docs_response.status_code == 200

            docs_data = docs_response.json()
            assert "items" in docs_data
            assert "total" in docs_data
            assert "links" in docs_data
            assert docs_data["limit"] == 2
            assert docs_data["offset"] == 0

            # Verify pagination links structure
            pagination_links = {link["rel"]: link for link in docs_data["links"]}
            assert "self" in pagination_links
            assert "limit=2" in pagination_links["self"]["href"]
            assert "offset=0" in pagination_links["self"]["href"]

            # Test 3: Test field selection on documents
            filtered_docs = client.get(f"/libraries/{library_id}/documents?fields=id,metadata&limit=5")
            assert filtered_docs.status_code == 200

            filtered_data = filtered_docs.json()
            if filtered_data["items"]:
                doc = filtered_data["items"][0]
                expected_fields = {"id", "metadata"}
                actual_fields = set(doc.keys())
                assert expected_fields.issubset(actual_fields)

            # Test 4: Test chunk pagination with field selection (exclude embeddings)
            chunks_response = client.get(f"/libraries/{library_id}/chunks?fields=id,text,metadata&limit=2")
            assert chunks_response.status_code == 200

            chunks_data = chunks_response.json()
            assert chunks_data["limit"] == 2

            if chunks_data["items"]:
                chunk = chunks_data["items"][0]
                assert "id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk
                assert "embedding" not in chunk  # Should be excluded by field selection

            # Test 5: Test async index building with 202 status
            index_request = {"index_type": "linear", "async_operation": True}
            async_response = client.post(f"/libraries/{library_id}/index", json=index_request)
            assert async_response.status_code == 202

            # Verify async response structure
            async_data = async_response.json()
            assert "operation_id" in async_data
            assert "status" in async_data
            assert "links" in async_data
            assert async_data["status"] == "pending"

            # Verify Location header
            assert "location" in async_response.headers
            location = async_response.headers["location"]
            assert "/api/operations/" in location

            # Test 6: Monitor async operation status
            operation_id = async_data["operation_id"]
            status_response = client.get(f"/api/operations/{operation_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            assert status_data["operation_id"] == operation_id
            assert status_data["operation_type"] == "index_build"
            assert status_data["resource_id"] == library_id

            # Wait for operation to complete or progress
            max_attempts = 5
            for attempt in range(max_attempts):
                time.sleep(0.3)
                status_response = client.get(f"/api/operations/{operation_id}")
                status_data = status_response.json()
                if status_data["status"] in ["completed", "failed"]:
                    break

            # Test 7: Test search with HATEOAS links after indexing
            search_response = client.get(f"/libraries/{library_id}/search?query_text=integration&top_k=3")
            assert search_response.status_code == 200

            search_data = search_response.json()
            assert "results" in search_data
            assert "links" in search_data

            # Verify search response HATEOAS links
            search_links = {link["rel"]: link for link in search_data["links"]}
            assert "self" in search_links
            assert "library" in search_links
            assert f"/libraries/{library_id}" in search_links["library"]["href"]

            # Test 8: Test library list with field selection and pagination
            libs_response = client.get("/libraries?fields=id,name,document_count&limit=10&offset=0")
            assert libs_response.status_code == 200

            libs_data = libs_response.json()
            assert "items" in libs_data
            assert "total" in libs_data
            assert libs_data["limit"] == 10

            if libs_data["items"]:
                lib = libs_data["items"][0]
                # Should have selected fields plus HATEOAS links
                expected_fields = {"id", "name", "document_count"}
                actual_fields = set(lib.keys())
                assert expected_fields.issubset(actual_fields)

                # Should have HATEOAS links even with field selection
                assert "links" in lib
                assert isinstance(lib["links"], list)

            # Test 9: Verify bandwidth savings with field selection
            full_chunks = client.get(f"/libraries/{library_id}/chunks")
            filtered_chunks = client.get(f"/libraries/{library_id}/chunks?fields=id,text")

            if full_chunks.status_code == 200 and filtered_chunks.status_code == 200:
                full_size = len(json.dumps(full_chunks.json()))
                filtered_size = len(json.dumps(filtered_chunks.json()))

                # Should have some bandwidth savings
                assert filtered_size <= full_size

        finally:
            # Cleanup
            client.delete(f"/libraries/{library_id}")

    def test_error_handling_with_api_features(self):
        """Test error handling maintains API best practices."""
        # Test pagination with invalid parameters
        response = client.get("/libraries?limit=200&offset=-1")
        assert response.status_code == 422  # Validation error

        # Test field selection with invalid fields
        lib_data = {"name": "Error Test", "metadata": {"description": "Test", "tags": []}}
        lib_response = client.post("/libraries", json=lib_data)
        library_id = lib_response.json()["id"]

        try:
            response = client.get(f"/libraries?fields=invalid_field,another_invalid")
            assert response.status_code == 400
            error_data = response.json()
            assert "Invalid fields" in error_data["detail"]

            # Test async operation on non-existent library
            fake_id = "00000000-0000-0000-0000-000000000000"
            response = client.post(f"/libraries/{fake_id}/index", json={"async_operation": True})
            assert response.status_code == 404

            # Test operation status for non-existent operation
            response = client.get(f"/api/operations/{fake_id}")
            assert response.status_code == 404

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_concurrent_operations_api_compliance(self):
        """Test concurrent operations maintain API best practices."""
        # Create library for concurrent testing
        lib_data = {"name": "Concurrent Test", "metadata": {"description": "Test", "tags": []}}
        lib_response = client.post("/libraries", json=lib_data)
        library_id = lib_response.json()["id"]

        try:
            # Add some test data
            doc_data = {
                "document_data": {"metadata": {"title": "Concurrent Test Doc"}},
                "chunk_texts": ["Concurrent test chunk."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Start multiple async operations
            operations = []
            for i in range(3):
                response = client.post(f"/libraries/{library_id}/index",
                                     json={"index_type": "linear", "async_operation": True})
                assert response.status_code == 202
                operations.append(response.json()["operation_id"])

            # Verify all operations are tracked
            for op_id in operations:
                status_response = client.get(f"/api/operations/{op_id}")
                assert status_response.status_code == 200

                status_data = status_response.json()
                assert "links" in status_data
                assert "status" in status_data["links"]
                assert "cancel" in status_data["links"]

            # Test that pagination works during operations
            chunks_response = client.get(f"/libraries/{library_id}/chunks?limit=5")
            assert chunks_response.status_code == 200

            # Should still have proper pagination structure
            chunks_data = chunks_response.json()
            assert "total" in chunks_data
            assert "limit" in chunks_data
            assert "has_next" in chunks_data

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_api_compliance_edge_cases(self):
        """Test edge cases while maintaining API compliance."""
        # Test empty collection pagination
        response = client.get("/libraries?limit=10&offset=1000")
        assert response.status_code == 200

        data = response.json()
        assert data["items"] == []
        assert data["has_next"] is False
        assert data["has_previous"] is True  # offset > 0

        # Test field selection with empty collections
        lib_data = {"name": "Empty Test", "metadata": {"description": "Empty", "tags": []}}
        lib_response = client.post("/libraries", json=lib_data)
        library_id = lib_response.json()["id"]

        try:
            # Test field selection on empty documents
            response = client.get(f"/libraries/{library_id}/documents?fields=id,metadata")
            assert response.status_code == 200

            data = response.json()
            assert data["items"] == []
            assert "total" in data
            assert "links" in data

            # Test async operation cancellation
            index_response = client.post(f"/libraries/{library_id}/index",
                                       json={"async_operation": True})
            if index_response.status_code == 202:
                operation_id = index_response.json()["operation_id"]

                # Try to cancel
                cancel_response = client.delete(f"/api/operations/{operation_id}")
                assert cancel_response.status_code in [204, 409]  # Success or already completed

        finally:
            client.delete(f"/libraries/{library_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])