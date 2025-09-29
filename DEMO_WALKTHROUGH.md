# Vector Database API - Real-Time Demo Walkthrough

## **Project Goal Achieved**

This demonstrates a **REST API that allows users to index and query their documents within a Vector Database** - specialized for storing vector embeddings and enabling fast similarity searches for NLP, recommendation systems, and more.

---

## **Part 1: Installation & Setup**

### **Step 1: Prerequisites Check**
```bash
# Verify Docker is installed
docker --version
# Should show: Docker version 20.10+

# Verify Git is installed
git --version
# Should show: git version 2.30+
```

### **Step 2: Get the Project**
```bash
# Clone the repository (or navigate to existing project)
cd /Users/dhruvmiyani/Downloads/Projects/IndexQueryAPI

# Verify project structure
ls -la
# You should see: Dockerfile, docker-compose.yml, src/, sdk/, requirements.txt
```

### **Step 3: Build the Vector Database**
```bash
# Build the Docker image (takes 2-3 minutes first time)
make build

# Alternative: docker build -t vectordb-api .
```

### **Step 4: Start the Vector Database**
```bash
# Start the containerized API
make run

# Alternative: docker run -d --name vectordb -p 8000:8000 -v vectordb_data:/app/data vectordb-api

# Verify it's running
docker ps
# Should show container named 'vectordb' running on port 8000
```

### **Step 5: Health Check**
```bash
# Test the API is responding
curl http://localhost:8000/health

# Expected output:
# {"status":"healthy","embedding_service":{"provider":"cohere","available":false,"dimension":1024},"features":{"vector_search":true,"multiple_indexes":true,"metadata_filtering":true,"batch_operations":true}}
```

---

## **Part 2: Real-Time Document Indexing & Querying**

### **Step 1: Access API Documentation**
```bash
# Open in browser for interactive API testing
open http://localhost:8000/docs
# Or visit: http://localhost:8000/docs
```

### **Step 2: Create Your First Document Library**
```bash
# Create a library for storing AI research papers
curl -X POST "http://localhost:8000/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Research Papers",
    "metadata": {
      "description": "Collection of artificial intelligence research documents",
      "domain": "AI/ML",
      "created_by": "demo_user"
    }
  }'

# Expected output: Library object with ID
# Save the "id" field for next steps (e.g., "550e8400-e29b-41d4-a716-446655440000")
```

### **Step 3: Add Documents with Text Chunks**
```bash
# Add a document about machine learning (replace LIBRARY_ID with actual ID)
export LIBRARY_ID="your-library-id-from-step-2"

curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "document_data": {
      "metadata": {
        "title": "Introduction to Machine Learning",
        "author": "Dr. Smith",
        "year": 2023,
        "category": "ML"
      }
    },
    "chunk_texts": [
      "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
      "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
      "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data."
    ]
  }'

# Add another document about natural language processing
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "document_data": {
      "metadata": {
        "title": "Natural Language Processing Fundamentals",
        "author": "Dr. Johnson",
        "year": 2023,
        "category": "NLP"
      }
    },
    "chunk_texts": [
      "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
      "Transformers have revolutionized NLP by providing a way to process sequential data in parallel, leading to models like BERT and GPT.",
      "Text embeddings convert words and sentences into numerical vectors that capture semantic meaning and relationships."
    ]
  }'
```

### **Step 4: Build Vector Index for Fast Searching**
```bash
# Build an optimized vector index for similarity search
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/index" \
  -H "Content-Type: application/json" \
  -d '{
    "index_type": "optimized_linear",
    "force_rebuild": true
  }'

# Expected output: {"status": "success", "index_type": "optimized_linear", ...}
```

### **Step 5: Query Documents with Similarity Search**
```bash
# Search for content related to "neural networks"
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "neural networks and deep learning",
    "top_k": 3
  }' | python3 -m json.tool

# Search for NLP-related content
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "language processing and text analysis",
    "top_k": 3
  }' | python3 -m json.tool

# Search with metadata filters (only 2023 papers)
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "artificial intelligence",
    "top_k": 5,
    "filters": {"year": 2023}
  }' | python3 -m json.tool
```

---

## **Part 3: Advanced Features Demo**

### **Test Different Index Types**
```bash
# Test with HNSW index (best for large datasets)
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/index" \
  -H "Content-Type: application/json" \
  -d '{
    "index_type": "hnsw",
    "force_rebuild": true
  }'

# Test with LSH index (good for approximate search)
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/index" \
  -H "Content-Type: application/json" \
  -d '{
    "index_type": "multiprobe_lsh",
    "force_rebuild": true
  }'
```

### **Complex Metadata Filtering**
```bash
# Search only in ML category
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "learning algorithms",
    "top_k": 3,
    "filters": {"category": "ML"}
  }' | python3 -m json.tool

# Search by author
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "artificial intelligence",
    "top_k": 3,
    "filters": {"author": "Dr. Smith"}
  }' | python3 -m json.tool
```

### **Check Index Statistics**
```bash
# Get index performance stats
curl "http://localhost:8000/libraries/$LIBRARY_ID/index/stats" | python3 -m json.tool
```

---

## **Part 4: Python SDK Demo**

### **Using the Python Client**
```bash
# Run the SDK example
docker exec vectordb python3 /app/sdk/examples.py

# Or run locally if you have Python:
cd sdk
python3 examples.py
```

### **Custom Python Script**
```python
# Create a custom script: demo_script.py
from sdk.vectordb_client import VectorDBClient, IndexType
import json

# Initialize client
client = VectorDBClient("http://localhost:8000")

# Create library
library = client.create_library(
    name="Demo Library",
    description="Real-time demo library"
)
print(f"Created library: {library.name} (ID: {library.id})")

# Add document
doc = client.create_document(
    library_id=library.id,
    title="AI Demo Document",
    chunks=[
        {"text": "Artificial intelligence is transforming industries worldwide."},
        {"text": "Machine learning enables systems to improve automatically."}
    ],
    metadata={"author": "Demo User", "type": "educational"}
)
print(f"Added document with {doc.chunk_count} chunks")

# Build index
result = client.build_index(library.id, IndexType.OPTIMIZED_LINEAR)
print(f"Built index: {result}")

# Search
results = client.search(
    library_id=library.id,
    query_text="AI and automation",
    top_k=2
)
print("Search results:")
for i, result in enumerate(results, 1):
    print(f"  {i}. Score: {result.score:.3f}")
    print(f"     Text: {result.text[:100]}...")

# Cleanup
client.delete_library(library.id)
print("Demo completed successfully!")
```

---

## **Part 5: Real-Time Monitoring**

### **Monitor Container Performance**
```bash
# Watch resource usage
docker stats vectordb

# View live logs
docker logs -f vectordb

# Check health status
curl http://localhost:8000/health | python3 -m json.tool
```

### **Database Statistics**
```bash
# Get overall database stats
curl http://localhost:8000/statistics | python3 -m json.tool

# List all libraries
curl http://localhost:8000/libraries | python3 -m json.tool
```

---

## **Part 6: Use Cases Demonstrated**

### **1. Document Similarity Search**
**Demonstrated**: Semantic search across research papers
**Use Case**: Knowledge management, document discovery

### **2. Content Recommendation**
**Demonstrated**: Finding related content based on queries
**Use Case**: Content recommendation engines

### **3. Semantic Search**
**Demonstrated**: Finding conceptually similar text, not just keyword matches
**Use Case**: Search engines, chatbots, Q&A systems

### **4. Metadata Filtering**
**Demonstrated**: Filtering by author, year, category
**Use Case**: Faceted search, content categorization

---

## **Cleanup**

```bash
# Stop and remove container
make remove

# Or manually:
docker stop vectordb
docker rm vectordb

# Remove image
docker rmi vectordb-api

# Remove data volume (optional)
docker volume rm vectordb_data
```

---

## **Success! You've Successfully:**

**Installed** a containerized Vector Database API
**Indexed** documents with text chunks
**Queried** using semantic similarity search
**Filtered** results by metadata
**Tested** multiple index algorithms
**Used** the Python SDK
**Monitored** real-time performance

**Your Vector Database is now ready for production use in NLP, recommendation systems, and any application requiring fast similarity search!**

## **For Screen Recording**

This walkthrough is designed to be recorded as a demo video. Each step shows clear input/output and demonstrates the core functionality:
- Document indexing
- Vector similarity search
- Metadata filtering
- Real-time performance
- Multiple index algorithms
- Python SDK usage