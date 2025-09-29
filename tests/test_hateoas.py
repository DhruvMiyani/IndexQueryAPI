"""
Tests for RESTful API best practice: HATEOAS (Hypermedia as Engine of Application State).

Tests hypermedia links in API responses for navigation and discoverability.
"""

import pytest
from fastapi.testclient import TestClient
import json

from main import app

client = TestClient(app)


class TestHATEOAS:
    """Test HATEOAS implementation in API responses."""

    def test_library_response_contains_links(self):
        """Test that library responses contain hypermedia links."""
        # Create test library
        library_data = {
            "name": "HATEOAS Test Library",
            "metadata": {
                "description": "Library for testing HATEOAS links",
                "tags": ["hateoas", "test"]
            }
        }
        create_response = client.post("/libraries", json=library_data)
        assert create_response.status_code == 201
        library_id = create_response.json()["id"]

        try:
            # Get library list - should contain links
            response = client.get("/libraries")
            assert response.status_code == 200

            data = response.json()
            assert "items" in data

            if data["items"]:
                library = data["items"][0]

                # Should have links array
                assert "links" in library
                assert isinstance(library["links"], list)

                # Convert links to dictionary for easier testing
                links_dict = {link["rel"]: link for link in library["links"]}

                # Should have standard library links
                expected_rels = ["self", "edit", "delete", "documents", "chunks", "index", "search"]
                for rel in expected_rels:
                    assert rel in links_dict, f"Missing '{rel}' link"

                # Verify link structure
                self_link = links_dict["self"]
                assert "href" in self_link
                assert "method" in self_link
                assert self_link["method"] == "GET"
                assert f"/libraries/{library['id']}" in self_link["href"]

                # Verify edit link
                edit_link = links_dict["edit"]
                assert edit_link["method"] == "PUT"
                assert f"/libraries/{library['id']}" in edit_link["href"]

                # Verify delete link
                delete_link = links_dict["delete"]
                assert delete_link["method"] == "DELETE"

                # Verify related resource links
                docs_link = links_dict["documents"]
                assert docs_link["method"] == "GET"
                assert f"/libraries/{library['id']}/documents" in docs_link["href"]

                chunks_link = links_dict["chunks"]
                assert f"/libraries/{library['id']}/chunks" in chunks_link["href"]

                search_link = links_dict["search"]
                assert f"/libraries/{library['id']}/search" in search_link["href"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_pagination_contains_navigation_links(self):
        """Test that paginated responses contain navigation links."""
        # Create multiple libraries for pagination testing
        library_ids = []
        try:
            for i in range(3):
                library_data = {
                    "name": f"Pagination Test Library {i}",
                    "metadata": {"description": f"Library {i}", "tags": []}
                }
                response = client.post("/libraries", json=library_data)
                library_ids.append(response.json()["id"])

            # Get paginated results
            response = client.get("/libraries?limit=2&offset=0")
            assert response.status_code == 200

            data = response.json()

            # Pagination response should have links
            assert "links" in data
            assert isinstance(data["links"], list)

            # Convert to dict for easier testing
            links_dict = {link["rel"]: link for link in data["links"]}

            # Should have self link
            assert "self" in links_dict
            self_link = links_dict["self"]
            assert "limit=2" in self_link["href"]
            assert "offset=0" in self_link["href"]

            # If there are more items, should have next link
            if data["has_next"]:
                assert "next" in links_dict
                next_link = links_dict["next"]
                assert "offset=2" in next_link["href"]

            # Should have first link
            if "first" in links_dict:
                first_link = links_dict["first"]
                assert "offset=0" in first_link["href"]

        finally:
            for library_id in library_ids:
                client.delete(f"/libraries/{library_id}")

    def test_document_response_contains_links(self):
        """Test that document responses contain proper hypermedia links."""
        # Create library
        library_data = {
            "name": "Document HATEOAS Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Create document
            doc_data = {
                "document_data": {
                    "metadata": {
                        "title": "HATEOAS Test Document",
                        "author": "Test Author"
                    }
                },
                "chunk_texts": ["Test chunk for HATEOAS links."]
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=doc_data)
            assert doc_response.status_code == 201

            # Get documents list
            response = client.get(f"/libraries/{library_id}/documents")
            assert response.status_code == 200

            data = response.json()
            if data["items"]:
                document = data["items"][0]

                # Should have links
                assert "links" in document
                assert isinstance(document["links"], list)

                links_dict = {link["rel"]: link for link in document["links"]}

                # Should have standard document links
                expected_rels = ["self", "edit", "delete", "library", "chunks"]
                for rel in expected_rels:
                    assert rel in links_dict, f"Missing '{rel}' link in document"

                # Verify library link points back to parent
                library_link = links_dict["library"]
                assert f"/libraries/{library_id}" in library_link["href"]

                # Verify chunks link
                chunks_link = links_dict["chunks"]
                assert f"/libraries/{library_id}/chunks" in chunks_link["href"]
                assert f"document_id={document['id']}" in chunks_link["href"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_chunk_response_contains_links(self):
        """Test that chunk responses contain proper hypermedia links."""
        # Create library and document with chunks
        library_data = {
            "name": "Chunk HATEOAS Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            doc_data = {
                "document_data": {
                    "metadata": {"title": "HATEOAS Chunk Test Doc"}
                },
                "chunk_texts": ["HATEOAS test chunk content."]
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=doc_data)
            document_id = doc_response.json()["id"]

            # Get chunks
            response = client.get(f"/libraries/{library_id}/chunks")
            assert response.status_code == 200

            data = response.json()
            if data["items"]:
                chunk = data["items"][0]

                # Should have links
                assert "links" in chunk
                assert isinstance(chunk["links"], list)

                links_dict = {link["rel"]: link for link in chunk["links"]}

                # Should have standard chunk links
                expected_rels = ["self", "edit", "delete", "library", "document"]
                for rel in expected_rels:
                    assert rel in links_dict, f"Missing '{rel}' link in chunk"

                # Verify library link
                library_link = links_dict["library"]
                assert f"/libraries/{library_id}" in library_link["href"]

                # Verify document link
                document_link = links_dict["document"]
                assert f"/libraries/{library_id}/documents/{document_id}" in document_link["href"]

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_operation_response_contains_links(self):
        """Test that async operation responses contain proper links."""
        # Create library
        library_data = {
            "name": "Operation HATEOAS Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add some data to index
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Operation Test Doc"}
                },
                "chunk_texts": ["Operation test chunk."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Start async operation
            index_request = {
                "index_type": "linear",
                "async_operation": True
            }
            response = client.post(f"/libraries/{library_id}/index", json=index_request)
            assert response.status_code == 202

            data = response.json()

            # Should have links
            assert "links" in data
            assert isinstance(data["links"], dict)

            links = data["links"]

            # Should have status and cancel links
            assert "status" in links
            assert "cancel" in links

            # Status link should point to operation endpoint
            status_url = links["status"]
            assert "/api/operations/" in status_url
            assert data["operation_id"] in status_url

            # Cancel link should be DELETE method to same endpoint
            cancel_url = links["cancel"]
            assert "/api/operations/" in cancel_url
            assert data["operation_id"] in cancel_url

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_search_response_contains_links(self):
        """Test that search responses contain proper links."""
        # Create library with searchable content
        library_data = {
            "name": "Search HATEOAS Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Add searchable content
            doc_data = {
                "document_data": {
                    "metadata": {"title": "Searchable Document"}
                },
                "chunk_texts": ["This is searchable content for HATEOAS testing."]
            }
            client.post(f"/libraries/{library_id}/documents", json=doc_data)

            # Build index first
            index_request = {"index_type": "linear", "force_rebuild": True}
            client.post(f"/libraries/{library_id}/index", json=index_request)

            # Perform search
            response = client.get(f"/libraries/{library_id}/search?query_text=searchable&top_k=5")
            assert response.status_code == 200

            data = response.json()

            # Search response should have links
            assert "links" in data
            assert isinstance(data["links"], list)

            links_dict = {link["rel"]: link for link in data["links"]}

            # Should have self link
            assert "self" in links_dict
            self_link = links_dict["self"]
            assert f"/libraries/{library_id}/search" in self_link["href"]

            # Should have library link
            assert "library" in links_dict
            library_link = links_dict["library"]
            assert f"/libraries/{library_id}" in library_link["href"]

            # May have post-search link for advanced search
            if "post-search" in links_dict:
                post_search_link = links_dict["post-search"]
                assert post_search_link["method"] == "POST"

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_links_have_proper_structure(self):
        """Test that all links follow the proper structure."""
        library_data = {
            "name": "Link Structure Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            response = client.get("/libraries")
            data = response.json()

            if data["items"]:
                library = data["items"][0]
                links = library["links"]

                for link in links:
                    # Each link should have required fields
                    assert "rel" in link
                    assert "href" in link
                    assert "method" in link

                    # rel should be a string
                    assert isinstance(link["rel"], str)
                    assert len(link["rel"]) > 0

                    # href should be a valid-looking URL
                    assert isinstance(link["href"], str)
                    assert link["href"].startswith("/") or link["href"].startswith("http")

                    # method should be valid HTTP method
                    valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
                    assert link["method"] in valid_methods

                    # Optional fields
                    if "type" in link:
                        assert isinstance(link["type"], str)

                    if "title" in link:
                        assert isinstance(link["title"], str)

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_links_survive_field_selection(self):
        """Test that HATEOAS links are preserved with field selection."""
        library_data = {
            "name": "Field Selection Links Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Get with field selection
            response = client.get("/libraries?fields=id,name,links")
            assert response.status_code == 200

            data = response.json()
            if data["items"]:
                library = data["items"][0]

                # Should still have links even with field selection
                assert "links" in library
                assert isinstance(library["links"], list)
                assert len(library["links"]) > 0

                # Should have selected fields
                assert "id" in library
                assert "name" in library

        finally:
            client.delete(f"/libraries/{library_id}")

    def test_link_urls_are_functional(self):
        """Test that the URLs in HATEOAS links actually work."""
        library_data = {
            "name": "Functional Links Test",
            "metadata": {"description": "Test", "tags": []}
        }
        lib_response = client.post("/libraries", json=library_data)
        library_id = lib_response.json()["id"]

        try:
            # Get library with links
            response = client.get("/libraries")
            data = response.json()

            if data["items"]:
                library = data["items"][0]
                links_dict = {link["rel"]: link for link in library["links"]}

                # Test that self link works
                if "self" in links_dict:
                    self_link = links_dict["self"]
                    # Extract just the path part
                    href = self_link["href"]
                    if href.startswith("http"):
                        # Extract path from full URL
                        from urllib.parse import urlparse
                        href = urlparse(href).path

                    # Test the link
                    link_response = client.get(href)
                    assert link_response.status_code == 200

                # Test documents link
                if "documents" in links_dict:
                    docs_link = links_dict["documents"]
                    href = docs_link["href"]
                    if href.startswith("http"):
                        from urllib.parse import urlparse
                        href = urlparse(href).path

                    docs_response = client.get(href)
                    assert docs_response.status_code == 200

        finally:
            client.delete(f"/libraries/{library_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])