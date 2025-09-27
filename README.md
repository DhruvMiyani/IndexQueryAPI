# Vector Database REST API

A high-performance vector database REST API built with FastAPI, featuring custom-implemented indexing algorithms for k-nearest neighbor search. Designed with Clean Code principles, Domain-Driven Design (DDD), and SOLID principles for maintainable, scalable software architecture.

## Features

- 🚀 **Custom Vector Indexing**: Three indexing algorithms implemented from scratch (Linear, KD-Tree, LSH)
- 📚 **Library Management**: Create and manage collections of documents and chunks
- 🔍 **Semantic Search**: k-Nearest Neighbor vector similarity search with cosine similarity
- 🎯 **Metadata Filtering**: Filter search results by metadata attributes
- 🐳 **Production-Ready Docker**: Multi-stage builds, health checks, security best practices
- 📝 **Auto Documentation**: Interactive API docs with Swagger UI at `/docs`
- ✅ **Comprehensive Testing**: Unit tests, integration tests, and Docker deployment tests
- 🔧 **Clean Code Implementation**: SOLID principles, early returns, meaningful names
- 🛠️ **Development Automation**: Comprehensive Makefile for streamlined workflows
- 🌐 **Cohere Integration**: 1024-dimensional embeddings for high-quality semantic search
- 🔒 **Thread-Safe**: Concurrent request handling with proper asyncio locking
- 📊 **Index Performance**: Configurable algorithms optimized for different data sizes and dimensions

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, but recommended)
- Make (for automated commands)
- Cohere API key (provided in project)

### Installation Options

#### Option 1: Automated Setup with Make (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd stackai.vectorapi

# Quick start - installs dependencies and runs the application
make quickstart

# The application will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

#### Option 2: Local Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd stackai.vectorapi

# Install Python dependencies
make install

# Set up development environment with additional tools
make dev-install

# Run the application locally
make run

# Or run with specific API key
COHERE_API_KEY=pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd make run
```

#### Option 3: Docker Development
```bash
# Build and run with Docker Compose (recommended for development)
make docker-up

# Or build and run individual container
make docker-build
make docker-run

# View logs
make docker-logs
```

#### Option 4: Production Docker Deployment
```bash
# Build production image
docker build -t vector-db-api .

# Run production container with health checks
docker run -d -p 8000:8000 \
  -e COHERE_API_KEY=pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd \
  --name vector-api \
  --restart unless-stopped \
  vector-db-api

# Check health status
curl http://localhost:8000/health
```

#### Option 5: Manual Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export COHERE_API_KEY=pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd

# Run the application
cd src
python main.py
```

## Manual Testing Guide - Complete Flow

Once the application is running, navigate to **http://localhost:8000/docs** to access the interactive Swagger UI.

### Step 1: Verify API Health
1. Click on **GET /health**
2. Click **"Try it out"**
3. Click **"Execute"**
4. Verify the response shows:
   ```json
   {
     "status": "healthy",
     "embedding_service": {
       "provider": "cohere",
       "available": true,
       "dimension": 1024
     }
   }
   ```

### Step 2: Create a Library
1. Click on **POST /libraries**
2. Click **"Try it out"**
3. Replace the request body with:
   ```json
   {
     "name": "AI Research Library",
     "description": "Collection of AI and ML documents",
     "index_type": "linear",
     "metadata": {
       "category": "research",
       "owner": "test_user"
     }
   }
   ```
4. Click **"Execute"**
5. Copy the `library_id` from the response (you'll need it for next steps)

Example response:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "AI Research Library",
  "description": "Collection of AI and ML documents",
  "index_type": "linear",
  "metadata": {
    "category": "research",
    "owner": "test_user"
  },
  "created_at": "2024-01-20T10:00:00Z",
  "updated_at": "2024-01-20T10:00:00Z"
}
```

### Step 3: Create a Document
1. Click on **POST /libraries/{library_id}/documents**
2. Click **"Try it out"**
3. Enter your `library_id` in the path parameter
4. Replace the request body with:
   ```json
   {
     "name": "Introduction to Machine Learning",
     "metadata": {
       "author": "John Doe",
       "year": 2024,
       "topic": "ML Basics"
     }
   }
   ```
5. Click **"Execute"**
6. Copy the `document_id` from the response

### Step 4: Add Chunks with Text (Embeddings Auto-Generated)
1. Click on **POST /libraries/{library_id}/chunks**
2. Click **"Try it out"**
3. Enter your `library_id` in the path parameter
4. Add multiple chunks with different texts:

**Chunk 1:**
```json
{
  "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
  "document_id": "your-document-id-here",
  "metadata": {
    "chapter": 1,
    "section": "Introduction"
  }
}
```

**Chunk 2:**
```json
{
  "text": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. They are the foundation of deep learning.",
  "document_id": "your-document-id-here",
  "metadata": {
    "chapter": 2,
    "section": "Neural Networks"
  }
}
```

**Chunk 3:**
```json
{
  "text": "Natural language processing is a field of AI that gives machines the ability to read, understand, and derive meaning from human language.",
  "document_id": "your-document-id-here",
  "metadata": {
    "chapter": 3,
    "section": "NLP"
  }
}
```

5. Execute each chunk creation separately
6. The API will automatically generate embeddings using Cohere

### Step 5: Build an Index (Optional but Recommended)
1. Click on **POST /libraries/{library_id}/index**
2. Click **"Try it out"**
3. Enter your `library_id` in the path parameter
4. Choose an index type in the request body:
   ```json
   {
     "index_type": "kd_tree"
   }
   ```
   Options: `"linear"`, `"kd_tree"`, or `"lsh"`
5. Click **"Execute"**

### Step 6: Perform Semantic Search
1. Click on **POST /libraries/{library_id}/search**
2. Click **"Try it out"**
3. Enter your `library_id` in the path parameter
4. Try different search queries:

**Example 1 - Text Query:**
```json
{
  "query_text": "What is deep learning?",
  "top_k": 5,
  "include_text": true,
  "include_metadata": true
}
```

**Example 2 - Semantic Similarity:**
```json
{
  "query_text": "artificial intelligence and neural computation",
  "top_k": 3,
  "include_text": true,
  "min_score": 0.5
}
```

**Example 3 - With Metadata Filters:**
```json
{
  "query_text": "machine learning fundamentals",
  "top_k": 5,
  "filters": {
    "chapter": 1
  },
  "include_text": true,
  "include_metadata": true
}
```

5. Click **"Execute"**
6. Observe the results sorted by similarity score

Expected response structure:
```json
{
  "query": "What is deep learning?",
  "results": [
    {
      "chunk_id": "chunk-uuid",
      "document_id": "doc-uuid",
      "score": 0.8567,
      "text": "Neural networks are computing systems...",
      "metadata": {
        "chapter": 2,
        "section": "Neural Networks"
      }
    }
  ],
  "total_results": 1,
  "search_time_ms": 12.5,
  "index_type": "kd_tree"
}
```

### Step 7: Test Different Index Types
1. Go back to **POST /libraries/{library_id}/index**
2. Rebuild the index with different algorithms:
   - `"linear"` - Brute force search (most accurate)
   - `"kd_tree"` - Tree-based spatial index (fast for low dimensions)
   - `"lsh"` - Locality-Sensitive Hashing (fast approximate search)
3. Repeat the search queries and compare:
   - Search times
   - Result accuracy
   - Similarity scores

### Step 8: Test CRUD Operations

#### Update a Chunk:
1. Click on **PUT /libraries/{library_id}/chunks/{chunk_id}**
2. Update the text or metadata
3. The embedding will be automatically regenerated

#### Delete a Chunk:
1. Click on **DELETE /libraries/{library_id}/chunks/{chunk_id}**
2. Verify deletion with **GET /libraries/{library_id}/chunks**

#### List All Libraries:
1. Click on **GET /libraries**
2. See all created libraries with pagination support

### Step 9: Advanced Testing Scenarios

#### Test Bulk Chunk Creation:
1. Click on **POST /libraries/{library_id}/chunks/bulk**
2. Provide multiple texts at once:
```json
{
  "document_id": "your-document-id",
  "texts": [
    "First chunk text about machine learning",
    "Second chunk text about deep learning",
    "Third chunk text about neural networks"
  ]
}
```

#### Test Search Without Text (Direct Vector):
1. First, get an embedding from a chunk
2. Use **POST /libraries/{library_id}/search** with:
```json
{
  "query_vector": [0.123, -0.456, 0.789, ...],
  "top_k": 5
}
```

### Step 10: Performance Testing

1. Create a library with many chunks (50+)
2. Compare search performance:
   - Without index (linear scan)
   - With KD-Tree index
   - With LSH index
3. Monitor response times in the Swagger UI

### Expected Results

✅ **Successful Library Creation**: Returns library with unique ID
✅ **Chunk Creation**: Automatically generates 1024-dimensional embeddings
✅ **Semantic Search**: Returns relevant chunks even with different wording
✅ **Similarity Scores**: Range from 0.0 to 1.0 (higher is more similar)
✅ **Index Building**: Improves search performance for large datasets
✅ **Metadata Filtering**: Correctly filters results based on criteria

### Common Test Cases

1. **Semantic Similarity Test**:
   - Add chunk: "Machine learning is a type of artificial intelligence"
   - Search: "AI and ML technologies"
   - Expected: High similarity score (>0.6)

2. **Different Topics Test**:
   - Add chunk: "Python is a programming language"
   - Search: "Neural networks and deep learning"
   - Expected: Low similarity score (<0.3)

3. **Exact Match Test**:
   - Add chunk with specific text
   - Search with same text
   - Expected: Very high similarity score (>0.95)

### Troubleshooting

- **No results returned**: Check if chunks were added to the correct library
- **Low similarity scores**: Ensure Cohere API key is set correctly
- **Index not improving performance**: May need more data (>100 chunks) to see benefits
- **API errors**: Check the response details in Swagger UI for specific error messages

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API info |
| GET | `/health` | Health check with service status |
| POST | `/libraries` | Create a new library |
| GET | `/libraries` | List all libraries |
| GET | `/libraries/{id}` | Get library details |
| PUT | `/libraries/{id}` | Update library metadata |
| DELETE | `/libraries/{id}` | Delete a library |
| POST | `/libraries/{id}/documents` | Create a document |
| GET | `/libraries/{id}/documents/{doc_id}` | Get document details |
| POST | `/libraries/{id}/chunks` | Add a chunk |
| POST | `/libraries/{id}/chunks/bulk` | Add multiple chunks |
| GET | `/libraries/{id}/chunks` | List chunks in library |
| POST | `/libraries/{id}/index` | Build/rebuild index |
| POST | `/libraries/{id}/search` | Perform vector search |

## Testing with cURL

If you prefer command-line testing:

```bash
# Create a library
curl -X POST "http://localhost:8000/libraries" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Library", "index_type": "linear"}'

# Add a chunk (replace library-id and document-id)
curl -X POST "http://localhost:8000/libraries/{library-id}/chunks" \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample text", "document_id": "{document-id}"}'

# Search
curl -X POST "http://localhost:8000/libraries/{library-id}/search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Sample query", "top_k": 5}'
```

## Development Workflow

### Code Quality Tools
```bash
# Format code with Black
make format

# Lint code with Ruff
make lint

# Type checking with Pyright
make typecheck

# Run all quality checks
make check-all
```

### Testing Commands
```bash
# Run all tests
make test

# Run tests with coverage report
make test-coverage

# Run integration tests only
make test-integration

# Run specific test file
pytest tests/test_api_integration.py -v

# Test Docker deployment
make test-docker
```

### Docker Commands
```bash
# Development with hot reload
make docker-dev

# Production deployment
make docker-up

# View application logs
make docker-logs

# Clean up containers and images
make docker-clean

# Full rebuild and restart
make docker-rebuild
```

### API Testing Commands
```bash
# Test API health endpoint
make api-health

# Run complete API test flow
make api-test-flow

# Performance testing
make api-performance
```

## Architecture & Design

### Clean Code Implementation
This project implements the **10 Bulletproof Rules for Writing Clean Code**:

1. **Meaningful Names**: All variables, functions, and classes use descriptive names
   ```python
   # Good: Descriptive and clear
   similarity_score = calculate_cosine_similarity(query_vector, chunk_embedding)

   # Avoided: Vague and unclear
   s = calc(q, c)
   ```

2. **Single Responsibility Principle**: Each function and class has one clear purpose
   ```python
   class LinearIndex:
       """Handles only linear search operations"""

   class EmbeddingService:
       """Handles only text-to-vector conversion"""
   ```

3. **Self-Documenting Code**: Code is written to be understood without excessive comments
4. **Early Returns**: Functions return early on error conditions to reduce nesting
5. **No Hardcoded Values**: All constants are defined in configuration files
6. **Short Functions**: Maximum 20 lines per function for clarity
7. **Composition Over Inheritance**: Using dependency injection and composition patterns

### Domain-Driven Design (DDD) Architecture
```
src/
├── models/          # Domain entities (Chunk, Document, Library)
├── repository/      # Data access layer with abstract interfaces
├── services/        # Business logic and orchestration
├── api/            # FastAPI endpoints and HTTP handling
├── indexes/        # Vector search algorithms
├── core/           # Shared utilities (locks, embeddings, config)
└── tests/          # Comprehensive test suite
```

### SOLID Principles Implementation
- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: New index algorithms can be added without modifying existing code
- **Liskov Substitution**: All index implementations are interchangeable
- **Interface Segregation**: Small, focused interfaces rather than large ones
- **Dependency Inversion**: High-level modules depend on abstractions, not concretions

### Vector Index Algorithms

#### 1. Linear Index (Brute Force)
- **Time Complexity**: O(n·d) per query
- **Space Complexity**: O(n·d)
- **Use Case**: Small datasets, guaranteed exact results
- **Pros**: Simple, exact results, no build time
- **Cons**: Slow for large datasets

#### 2. KD-Tree Index
- **Build Time**: O(n log n)
- **Query Time**: O(log n) best case, O(n) worst case in high dimensions
- **Space Complexity**: O(n)
- **Use Case**: Low-to-moderate dimensions (< 20), exact search needed
- **Pros**: Fast for low dimensions, exact results
- **Cons**: Performance degrades in high dimensions (curse of dimensionality)

#### 3. LSH (Locality-Sensitive Hashing)
- **Query Time**: Sub-linear average case
- **Space Complexity**: O(n) plus hash tables
- **Use Case**: High dimensions, approximate results acceptable
- **Pros**: Scalable, fast for high-dimensional data
- **Cons**: Approximate results, tuning required

### Concurrency & Thread Safety
- **AsyncIO Integration**: Built for FastAPI's async context
- **Global Locking**: Prevents race conditions on shared in-memory data
- **Future Enhancement**: Read-write locks for better concurrent read performance

### Performance Characteristics
| Dataset Size | Recommended Index | Search Time | Build Time | Memory Usage |
|--------------|------------------|-------------|------------|--------------|
| < 1,000 chunks | Linear | ~1ms | None | Low |
| 1K-10K chunks | KD-Tree | ~0.1ms | ~100ms | Medium |
| > 10K chunks | LSH | ~0.01ms | ~200ms | High |

## Testing Strategy

### Test Coverage
- **Unit Tests**: Individual components (indexes, services, repositories)
- **Integration Tests**: Full API workflows with TestClient
- **Docker Tests**: Container deployment and health checks
- **Performance Tests**: Index algorithm benchmarking

### Test Data Strategy
- Small, predictable vectors for deterministic results
- Real-world text samples with Cohere embeddings
- Edge cases: empty libraries, invalid inputs, concurrent access

## Deployment

### Docker Features
- **Multi-stage build** for optimized production images
- **Non-root user** for security
- **Health checks** for container orchestration
- **Volume mounting** for data persistence
- **Environment variable** configuration
- **Graceful shutdown** handling

### Production Considerations
- Configure reverse proxy (nginx) for SSL termination
- Set up monitoring and logging
- Use Docker Compose or Kubernetes for orchestration
- Implement backup strategy for data persistence
- Configure rate limiting and authentication

## Performance Optimization

### Index Selection Guidelines
```python
# Automatic index recommendation
def recommend_index_type(dimension: int, dataset_size: int) -> IndexType:
    if dataset_size < 1000:
        return IndexType.LINEAR
    elif dimension <= 20:
        return IndexType.KD_TREE
    else:
        return IndexType.LSH
```

### Memory Management
- In-memory storage for fast access
- Configurable embedding dimensions
- Efficient vector operations with NumPy
- Lazy loading of large datasets

## API Documentation

Complete API documentation is available at `/docs` when the application is running. Key endpoints include:

- **Health Check**: `GET /health` - Service status and embedding provider info
- **Library Management**: Full CRUD operations for libraries
- **Document Management**: Create and manage document collections
- **Chunk Operations**: Add, update, delete text chunks with automatic embedding
- **Index Building**: `POST /libraries/{id}/index` - Build optimized search indexes
- **Vector Search**: `POST /libraries/{id}/search` - Semantic similarity search
- **Bulk Operations**: Efficient batch processing for large datasets

## Contributing

This project follows strict code quality standards:

1. **Code Style**: Black formatting, Ruff linting
2. **Type Checking**: Full static typing with Pyright
3. **Testing**: Comprehensive test coverage required
4. **Documentation**: Self-documenting code with minimal comments
5. **Architecture**: Follow DDD and SOLID principles

## License

This project is part of a technical assessment for Stack AI.