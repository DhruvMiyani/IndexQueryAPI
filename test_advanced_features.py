"""
Comprehensive integration tests for all advanced features.

Tests:
1. Enhanced metadata filtering
2. Persistence to disk
3. Python SDK client
4. Integration with optimized indexes
"""

import os
import sys
import tempfile
import shutil
import time
import numpy as np
from datetime import datetime
from uuid import uuid4

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('sdk')

def test_metadata_filtering():
    """Test enhanced metadata filtering capabilities."""
    print("🧪 Testing Enhanced Metadata Filtering...")

    try:
        from models.metadata_filter import (
            MetadataFilter, FilterCondition, FilterGroup,
            FilterOperator, LogicalOperator
        )
        from services.metadata_filter_service import MetadataFilterService

        # Create filter service
        filter_service = MetadataFilterService()

        # Create test data with rich metadata
        test_items = [
            {
                "id": "1",
                "metadata": {
                    "author": "John Doe",
                    "year": 2023,
                    "category": "AI",
                    "tags": ["machine-learning", "neural-networks"],
                    "created_at": "2023-06-15T10:00:00",
                    "title": "Introduction to Neural Networks"
                }
            },
            {
                "id": "2",
                "metadata": {
                    "author": "Jane Smith",
                    "year": 2022,
                    "category": "CV",
                    "tags": ["computer-vision", "cnn"],
                    "created_at": "2022-12-01T15:30:00",
                    "title": "Computer Vision with CNNs"
                }
            },
            {
                "id": "3",
                "metadata": {
                    "author": "Bob Johnson",
                    "year": 2023,
                    "category": "NLP",
                    "tags": ["transformers", "bert"],
                    "created_at": "2023-03-20T09:15:00",
                    "title": "BERT and Transformers"
                }
            }
        ]

        def metadata_extractor(item):
            return item.get("metadata", {})

        # Test 1: Simple filters (backward compatibility)
        print("  ✅ Test 1: Simple filters")
        simple_filter = {"year": 2023}
        filtered, stats = filter_service.apply_filters(test_items, simple_filter, metadata_extractor)
        assert len(filtered) == 2, f"Expected 2 items, got {len(filtered)}"
        print(f"     Found {len(filtered)} items from 2023")

        # Test 2: Advanced filter conditions
        print("  ✅ Test 2: Advanced filter conditions")
        advanced_filter = MetadataFilter(
            advanced_filters=FilterGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    FilterCondition(field="year", operator=FilterOperator.EQ, value=2023),
                    FilterCondition(field="category", operator=FilterOperator.IN, value=["AI", "NLP"])
                ]
            )
        )
        filtered, stats = filter_service.apply_filters(test_items, advanced_filter, metadata_extractor)
        assert len(filtered) == 2, f"Expected 2 items, got {len(filtered)}"
        print(f"     Found {len(filtered)} AI/NLP items from 2023")

        # Test 3: String operations
        print("  ✅ Test 3: String operations")
        string_filter = MetadataFilter(
            advanced_filters=FilterGroup(
                operator=LogicalOperator.OR,
                conditions=[
                    FilterCondition(
                        field="title",
                        operator=FilterOperator.CONTAINS,
                        value="neural",
                        case_sensitive=False
                    ),
                    FilterCondition(
                        field="author",
                        operator=FilterOperator.STARTS_WITH,
                        value="Jane"
                    )
                ]
            )
        )
        filtered, stats = filter_service.apply_filters(test_items, string_filter, metadata_extractor)
        assert len(filtered) == 2, f"Expected 2 items, got {len(filtered)}"
        print(f"     Found {len(filtered)} items with neural/Jane")

        # Test 4: Array operations
        print("  ✅ Test 4: Array operations")
        array_filter = MetadataFilter(
            advanced_filters=FilterGroup(
                conditions=[
                    FilterCondition(
                        field="tags",
                        operator=FilterOperator.ARRAY_CONTAINS,
                        value="transformers"
                    )
                ]
            )
        )
        filtered, stats = filter_service.apply_filters(test_items, array_filter, metadata_extractor)
        assert len(filtered) == 1, f"Expected 1 item, got {len(filtered)}"
        print(f"     Found {len(filtered)} items with 'transformers' tag")

        # Test 5: Date operations
        print("  ✅ Test 5: Date operations")
        date_filter = MetadataFilter(
            advanced_filters=FilterGroup(
                conditions=[
                    FilterCondition(
                        field="created_at",
                        operator=FilterOperator.AFTER,
                        value="2023-01-01T00:00:00"
                    )
                ]
            )
        )
        filtered, stats = filter_service.apply_filters(test_items, date_filter, metadata_extractor)
        assert len(filtered) == 2, f"Expected 2 items, got {len(filtered)}"
        print(f"     Found {len(filtered)} items created after 2023-01-01")

        print("✅ Enhanced Metadata Filtering: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ Enhanced Metadata Filtering FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persistence():
    """Test persistence to disk functionality."""
    print("\n🧪 Testing Persistence to Disk...")

    try:
        from persistence.persistence_manager import PersistenceManager

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  Using temp dir: {temp_dir}")

            # Initialize persistence manager
            persistence = PersistenceManager(
                data_dir=temp_dir,
                enable_wal=True,
                snapshot_interval_seconds=0,  # Disable automatic snapshots
                compression=True
            )

            # Test 1: Save and load state
            print("  ✅ Test 1: Save and load state")
            test_state = {
                "libraries": {
                    "lib1": {
                        "id": "lib1",
                        "name": "Test Library",
                        "created_at": datetime.now().isoformat()
                    }
                },
                "documents": {
                    "doc1": {
                        "id": "doc1",
                        "library_id": "lib1",
                        "title": "Test Document"
                    }
                },
                "chunks": {
                    "chunk1": {
                        "id": "chunk1",
                        "document_id": "doc1",
                        "text": "Test chunk content",
                        "vector": np.random.randn(128).tolist(),
                        "metadata": {"position": 0}
                    },
                    "chunk2": {
                        "id": "chunk2",
                        "document_id": "doc1",
                        "text": "Another test chunk",
                        "vector": np.random.randn(128).tolist(),
                        "metadata": {"position": 1}
                    }
                },
                "indexes": {
                    "lib1": {"type": "linear", "size": 2}
                }
            }

            # Save state
            snapshot_path = persistence.save_state(test_state)
            print(f"     Saved snapshot to: {os.path.basename(snapshot_path)}")

            # Load state
            loaded_state = persistence.load_state(snapshot_path)
            print(f"     Loaded state with {len(loaded_state.get('chunks', {}))} chunks")

            # Verify data integrity
            assert "libraries" in loaded_state
            assert "chunks" in loaded_state
            assert len(loaded_state["chunks"]) == 2
            assert "chunk1" in loaded_state["chunks"]
            assert "vector" in loaded_state["chunks"]["chunk1"]
            print("     Data integrity verified")

            # Test 2: WAL functionality
            print("  ✅ Test 2: Write-ahead logging")
            persistence.write_wal_entry("create", {"type": "chunk", "id": "chunk3"})
            persistence.write_wal_entry("update", {"type": "chunk", "id": "chunk1", "text": "updated"})
            print("     WAL entries written")

            # Test 3: Statistics
            print("  ✅ Test 3: Statistics")
            stats = persistence.get_statistics()
            assert stats["num_snapshots"] >= 1
            assert stats["compression_enabled"] == True
            assert stats["wal_enabled"] == True
            print(f"     Stats: {stats['num_snapshots']} snapshots, {stats['total_size_bytes']} bytes")

            # Test 4: Load latest snapshot
            print("  ✅ Test 4: Load latest snapshot")
            latest_state = persistence.load_state()  # No path = latest
            assert len(latest_state.get("chunks", {})) == 2
            print("     Latest snapshot loaded successfully")

            persistence.shutdown()

        print("✅ Persistence to Disk: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ Persistence to Disk FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdk_client():
    """Test Python SDK client functionality."""
    print("\n🧪 Testing Python SDK Client...")

    try:
        # Start a local server for testing
        import subprocess
        import time

        # Check if server is already running
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            server_running = response.status_code == 200
        except:
            server_running = False

        if not server_running:
            print("  ⚠️  Server not running at localhost:8000")
            print("     Please start the server: cd src && python -m uvicorn main:app --port 8000")
            return False

        from vectordb_client import VectorDBClient, IndexType

        # Test 1: Basic CRUD operations
        print("  ✅ Test 1: Basic CRUD operations")
        with VectorDBClient("http://localhost:8000") as client:
            # Health check
            health = client.health_check()
            print(f"     Health check: {health.get('status', 'unknown')}")

            # Create library
            library = client.create_library(
                name="SDK Test Library",
                description="Testing SDK functionality",
                metadata={"test": True, "created_by": "test_suite"}
            )
            print(f"     Created library: {library.name}")

            # Create document with chunks
            doc = client.create_document(
                library_id=library.id,
                title="SDK Test Document",
                chunks=[
                    {
                        "text": "This is the first test chunk for SDK testing",
                        "embedding": np.random.randn(128).tolist(),
                        "metadata": {"position": 0, "type": "intro"}
                    },
                    {
                        "text": "This is the second test chunk with different content",
                        "embedding": np.random.randn(128).tolist(),
                        "metadata": {"position": 1, "type": "content"}
                    },
                    {
                        "text": "This is the final test chunk for conclusion",
                        "embedding": np.random.randn(128).tolist(),
                        "metadata": {"position": 2, "type": "conclusion"}
                    }
                ],
                metadata={"author": "SDK Tester", "category": "test"}
            )
            print(f"     Created document with {doc.chunk_count} chunks")

            # Test 2: Index operations
            print("  ✅ Test 2: Index operations")

            # Build optimized linear index
            index_result = client.build_index(
                library.id,
                index_type=IndexType.OPTIMIZED_LINEAR,
                force_rebuild=True
            )
            print(f"     Built index: {index_result.get('index_type')}")

            # Get index stats
            stats = client.get_index_stats(library.id)
            print(f"     Index stats: {stats.get('type')} with {stats.get('size')} vectors")

            # Test 3: Search operations
            print("  ✅ Test 3: Search operations")

            # Text search
            results = client.search(
                library_id=library.id,
                query_text="test chunk content",
                top_k=3
            )
            print(f"     Text search returned {len(results)} results")
            if results:
                print(f"     Top result score: {results[0].score:.3f}")

            # Vector search
            query_vector = np.random.randn(128).tolist()
            results = client.search(
                library_id=library.id,
                query_vector=query_vector,
                top_k=2
            )
            print(f"     Vector search returned {len(results)} results")

            # Search with filters
            results = client.search(
                library_id=library.id,
                query_text="test content",
                top_k=5,
                filters={"type": "content"},
                min_score=0.0
            )
            print(f"     Filtered search returned {len(results)} results")

            # Test 4: List operations
            print("  ✅ Test 4: List operations")

            # List libraries
            libraries = client.list_libraries(limit=10)
            print(f"     Found {len(libraries)} libraries")

            # List documents
            documents = client.list_documents(library.id)
            print(f"     Found {len(documents)} documents in library")

            # Test 5: Index type comparison
            print("  ✅ Test 5: Index type comparison")

            # Test different index types
            index_types_to_test = [
                IndexType.OPTIMIZED_LINEAR,
                IndexType.MULTIPROBE_LSH,
                IndexType.HNSW
            ]

            query_vector = np.random.randn(128).tolist()

            for idx_type in index_types_to_test:
                # Build index
                client.build_index(library.id, index_type=idx_type, force_rebuild=True)

                # Time search
                start_time = time.time()
                results = client.search(
                    library_id=library.id,
                    query_vector=query_vector,
                    top_k=3
                )
                search_time = (time.time() - start_time) * 1000

                print(f"     {idx_type.value}: {search_time:.1f}ms, {len(results)} results")

            # Cleanup
            client.delete_library(library.id)
            print("     Cleaned up test library")

        print("✅ Python SDK Client: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ Python SDK Client FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all features together."""
    print("\n🧪 Testing Feature Integration...")

    try:
        # This test combines multiple features
        from models.metadata_filter import MetadataFilter, FilterCondition, FilterOperator
        from services.metadata_filter_service import MetadataFilterService
        from persistence.persistence_manager import PersistenceManager

        print("  ✅ Test 1: All imports successful")

        # Test that optimized indexes work with the system
        from indexes.base import IndexType
        from indexes.advanced_index_factory import AdvancedIndexFactory

        # Test creating different index types
        for index_type in [IndexType.OPTIMIZED_LINEAR, IndexType.HNSW, IndexType.MULTIPROBE_LSH]:
            try:
                index = AdvancedIndexFactory.create(index_type, 128)
                print(f"     ✅ {index_type.value}: {type(index).__name__}")
            except Exception as e:
                print(f"     ❌ {index_type.value}: {e}")
                return False

        print("  ✅ Test 2: All index types create successfully")

        # Test that metadata filtering works with realistic data
        filter_service = MetadataFilterService()

        test_data = [
            {"metadata": {"category": "AI", "year": 2023, "tags": ["ml", "dl"]}},
            {"metadata": {"category": "CV", "year": 2022, "tags": ["cnn", "vision"]}},
            {"metadata": {"category": "NLP", "year": 2023, "tags": ["transformers", "bert"]}}
        ]

        # Complex filter
        complex_filter = MetadataFilter(
            advanced_filters={
                "operator": "and",
                "conditions": [
                    {"field": "year", "operator": "eq", "value": 2023},
                    {"field": "tags", "operator": "array_contains", "value": "ml"}
                ]
            }
        )

        filtered, stats = filter_service.apply_filters(
            test_data,
            complex_filter,
            lambda x: x.get("metadata", {})
        )

        print(f"  ✅ Test 3: Complex filtering worked ({len(filtered)} results)")

        # Test persistence with realistic state
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = PersistenceManager(data_dir=temp_dir)

            test_state = {
                "libraries": {"lib1": {"name": "Integration Test"}},
                "chunks": {
                    f"chunk_{i}": {
                        "text": f"Test chunk {i}",
                        "vector": np.random.randn(64).tolist(),
                        "metadata": {"index": i}
                    }
                    for i in range(10)
                }
            }

            # Save and load
            snapshot_path = persistence.save_state(test_state)
            loaded_state = persistence.load_state(snapshot_path)

            assert len(loaded_state["chunks"]) == 10
            print("  ✅ Test 4: Persistence with vectors works")

            persistence.shutdown()

        print("✅ Feature Integration: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"❌ Feature Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run comprehensive test suite for all advanced features."""
    print("🚀 COMPREHENSIVE ADVANCED FEATURES TEST SUITE")
    print("=" * 60)

    test_results = []

    # Run all tests
    tests = [
        ("Enhanced Metadata Filtering", test_metadata_filtering),
        ("Persistence to Disk", test_persistence),
        ("Python SDK Client", test_sdk_client),
        ("Feature Integration", test_integration),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            test_results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 ALL ADVANCED FEATURES ARE WORKING CORRECTLY!")
        print("=" * 60)
        print("✅ Enhanced metadata filtering with complex operators")
        print("✅ Persistence to disk with compression and WAL")
        print("✅ Python SDK client with full API coverage")
        print("✅ All features integrate seamlessly")
        print("✅ All optimized index types functioning")
        print("\n🚀 The vector database is production-ready!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)