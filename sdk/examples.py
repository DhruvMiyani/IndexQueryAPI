"""
Example usage of Vector Database Python SDK Client.

Demonstrates common use cases and best practices.
"""

from vectordb_client import VectorDBClient, IndexType
import numpy as np


def basic_usage_example():
    """Basic CRUD operations example."""
    print("=== Basic Usage Example ===\n")

    # Initialize client
    client = VectorDBClient("http://localhost:8000")

    # Create a library
    library = client.create_library(
        name="My Knowledge Base",
        description="A collection of technical documents",
        metadata={"domain": "technology", "language": "en"},
    )
    print(f"Created library: {library.name} (ID: {library.id})")

    # Add a document with chunks
    doc = client.create_document(
        library_id=library.id,
        title="Introduction to AI",
        chunks=[
            {
                "text": "Artificial Intelligence (AI) is the simulation of human intelligence...",
                "embedding": np.random.randn(128).tolist(),  # Example embedding
            },
            {
                "text": "Machine learning is a subset of AI that enables systems to learn...",
                "embedding": np.random.randn(128).tolist(),
            },
            {
                "text": "Deep learning uses neural networks with multiple layers...",
                "embedding": np.random.randn(128).tolist(),
            },
        ],
        metadata={"author": "Dr. Smith", "year": 2023, "category": "AI"},
    )
    print(f"Created document: {doc.title} with {doc.chunk_count} chunks")

    # Build an index
    index_result = client.build_index(library.id, index_type=IndexType.OPTIMIZED_LINEAR)
    print(f"Built index: {index_result['index_type']}")

    # Search for similar content
    results = client.search(
        library_id=library.id,
        query_text="What is machine learning?",
        top_k=2,
    )

    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.3f}")
        if result.text:
            print(f"   Text: {result.text[:100]}...")

    # Clean up
    client.delete_library(library.id)
    print("\nLibrary deleted successfully")


def advanced_search_example():
    """Advanced search with filtering example."""
    print("\n=== Advanced Search Example ===\n")

    client = VectorDBClient("http://localhost:8000")

    # Create library with multiple documents
    library = client.create_library(name="Research Papers")

    # Add multiple documents with different metadata
    papers = [
        {
            "title": "Neural Networks in NLP",
            "chunks": [
                {"text": "Transformers revolutionized NLP...", "embedding": np.random.randn(128).tolist()},
                {"text": "BERT achieved state-of-the-art...", "embedding": np.random.randn(128).tolist()},
            ],
            "metadata": {
                "year": 2023,
                "category": "NLP",
                "author": "Alice",
                "tags": ["transformers", "bert", "nlp"],
            },
        },
        {
            "title": "Computer Vision Advances",
            "chunks": [
                {"text": "CNNs are fundamental for image processing...", "embedding": np.random.randn(128).tolist()},
                {"text": "Object detection has improved significantly...", "embedding": np.random.randn(128).tolist()},
            ],
            "metadata": {
                "year": 2022,
                "category": "CV",
                "author": "Bob",
                "tags": ["cnn", "object-detection", "cv"],
            },
        },
        {
            "title": "Reinforcement Learning",
            "chunks": [
                {"text": "Q-learning is a model-free algorithm...", "embedding": np.random.randn(128).tolist()},
                {"text": "Policy gradient methods optimize directly...", "embedding": np.random.randn(128).tolist()},
            ],
            "metadata": {
                "year": 2023,
                "category": "RL",
                "author": "Charlie",
                "tags": ["q-learning", "policy-gradient", "rl"],
            },
        },
    ]

    for paper in papers:
        client.create_document(
            library_id=library.id,
            title=paper["title"],
            chunks=paper["chunks"],
            metadata=paper["metadata"],
        )
        print(f"Added: {paper['title']}")

    # Build HNSW index for better performance
    client.build_index(library.id, index_type=IndexType.HNSW)
    print("\nBuilt HNSW index for fast search")

    # Search with metadata filters
    print("\n1. Search papers from 2023:")
    results = client.search(
        library_id=library.id,
        query_text="latest AI research",
        top_k=5,
        filters={"year": 2023},
    )
    for r in results:
        print(f"   - Score: {r.score:.3f}, Doc: {r.document_metadata.get('title') if r.document_metadata else 'N/A'}")

    print("\n2. Search NLP papers only:")
    results = client.search(
        library_id=library.id,
        query_text="language models",
        top_k=5,
        filters={"category": "NLP"},
    )
    for r in results:
        print(f"   - Score: {r.score:.3f}, Category: {r.document_metadata.get('category') if r.document_metadata else 'N/A'}")

    print("\n3. Search with minimum score threshold:")
    results = client.search(
        library_id=library.id,
        query_text="deep learning",
        top_k=10,
        min_score=0.5,
    )
    print(f"   Found {len(results)} results with score >= 0.5")

    # Clean up
    client.delete_library(library.id)


def batch_processing_example():
    """Example of batch processing large datasets."""
    print("\n=== Batch Processing Example ===\n")

    client = VectorDBClient("http://localhost:8000")

    # Create library
    library = client.create_library(
        name="Large Dataset",
        metadata={"processing_date": "2023-12-01"},
    )

    # Process in batches
    batch_size = 100
    num_batches = 5

    for batch_num in range(num_batches):
        chunks = []
        for i in range(batch_size):
            chunk_id = batch_num * batch_size + i
            chunks.append({
                "text": f"This is chunk {chunk_id} with some content about topic {chunk_id % 10}",
                "embedding": np.random.randn(128).tolist(),
                "metadata": {"batch": batch_num, "topic": f"topic_{chunk_id % 10}"}
            })

        # Create document for each batch
        doc = client.create_document(
            library_id=library.id,
            title=f"Batch {batch_num}",
            chunks=chunks,
            metadata={"batch_number": batch_num},
        )
        print(f"Processed batch {batch_num + 1}/{num_batches}: {len(chunks)} chunks")

    # Build index after all data is loaded
    print("\nBuilding IVF-PQ index for memory efficiency...")
    client.build_index(library.id, index_type=IndexType.IVF_PQ)

    # Get statistics
    stats = client.get_index_stats(library.id)
    print(f"Index stats: {stats}")

    # Perform searches
    print("\nSearching for topic-specific content...")
    results = client.search(
        library_id=library.id,
        query_text="information about topic 5",
        top_k=5,
        filters={"topic": "topic_5"},
    )
    print(f"Found {len(results)} relevant chunks for topic 5")

    # Clean up
    client.delete_library(library.id)


def context_manager_example():
    """Example using context manager for automatic cleanup."""
    print("\n=== Context Manager Example ===\n")

    # Using context manager ensures proper cleanup
    with VectorDBClient("http://localhost:8000") as client:
        # All operations within context
        library = client.create_library(name="Temporary Library")
        print(f"Working with library: {library.id}")

        # Add some data
        doc = client.create_document(
            library_id=library.id,
            title="Sample Document",
            chunks=[
                {"text": "Sample text", "embedding": np.random.randn(128).tolist()}
            ],
        )

        # Build index
        client.build_index(library.id, index_type=IndexType.OPTIMIZED_LINEAR)

        # Search
        results = client.search(
            library_id=library.id,
            query_vector=np.random.randn(128).tolist(),
            top_k=1,
        )
        print(f"Search completed: {len(results)} results")

        # Cleanup
        client.delete_library(library.id)

    print("Context manager closed - session cleaned up")


def index_comparison_example():
    """Compare different index types."""
    print("\n=== Index Comparison Example ===\n")

    client = VectorDBClient("http://localhost:8000")

    # Create test data
    library = client.create_library(name="Index Comparison")

    # Add substantial data for testing
    for i in range(10):
        chunks = [
            {
                "text": f"Document {i}, chunk {j}: {np.random.choice(['AI', 'ML', 'DL', 'NLP', 'CV'])} content",
                "embedding": np.random.randn(128).tolist(),
            }
            for j in range(50)
        ]

        client.create_document(
            library_id=library.id,
            title=f"Document {i}",
            chunks=chunks,
        )

    print(f"Added 500 chunks to library")

    # Test different index types
    index_types = [
        IndexType.OPTIMIZED_LINEAR,
        IndexType.MULTIPROBE_LSH,
        IndexType.HNSW,
    ]

    query_vector = np.random.randn(128).tolist()

    for index_type in index_types:
        print(f"\nTesting {index_type.value}:")

        # Build index
        result = client.build_index(
            library.id, index_type=index_type, force_rebuild=True
        )
        print(f"  Built: {result['index_type']}")

        # Get stats
        stats = client.get_index_stats(library.id)
        print(f"  Stats: Type={stats.get('type')}, Size={stats.get('size')}")

        # Perform search
        import time

        start_time = time.time()
        results = client.search(library_id=library.id, query_vector=query_vector, top_k=10)
        search_time = (time.time() - start_time) * 1000

        print(f"  Search time: {search_time:.1f}ms")
        print(f"  Results: {len(results)} found")
        if results:
            print(f"  Top score: {results[0].score:.3f}")

    # Clean up
    client.delete_library(library.id)


if __name__ == "__main__":
    # Run examples
    try:
        basic_usage_example()
        advanced_search_example()
        batch_processing_example()
        context_manager_example()
        index_comparison_example()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the API server is running at http://localhost:8000")