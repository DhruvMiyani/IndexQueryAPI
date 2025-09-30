# Vector Database REST API

A high-performance vector database REST API built with FastAPI

##  Quick Setup

### Prerequisites
- **Docker Engine 20.10+** and **Docker Compose 2.0+**
- **4GB+ RAM** recommended
- **2GB+ available disk space**

### 1. Build the Docker Image
```bash
make build
# Or: docker build -t vectordb-api:latest .
```

### 2. Start the Vector Database
```bash
make run
# Or: docker run -d --name vectordb -p 8000:8000 -v vectordb_data:/app/data vectordb-api:latest
```

### 3. Verify it's Running
```bash
# Check container status
docker ps

# Test health endpoint
curl http://localhost:8000/health
```

### 4. Access the API
- **API Base**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Quick API Test
```bash
# Create a library
curl -X POST "http://localhost:8000/libraries/" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Library", "metadata": {"description": "Demo library"}}'

# Note the library ID from response, then add a document
export LIBRARY_ID="your-library-id-here"

curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/documents/" \
  -H "Content-Type: application/json" \
  -d '{
    "document_data": {"metadata": {"title": "AI Document"}},
    "chunk_texts": ["Artificial intelligence is transforming technology."]
  }'

# Build index
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/index" \
  -H "Content-Type: application/json" \
  -d '{"index_type": "optimized_linear", "force_rebuild": true}'

# Search
curl -X POST "http://localhost:8000/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "machine learning", "top_k": 3}'
```

### 6. Access Swagger UI
- Open your browser
- Navigate to http://localhost:8000/docs
- You should see the interactive API documentation

### 7. Follow the Manual Testing Guide

**Step 7.1: Create a Library**
- Click on **POST /libraries**
- Use this JSON:
```json
{
  "name": "AI Research Library",
  "metadata": {
    "description": "Collection of AI and ML documents",
    "category": "research"
  }
}
```
- Copy the returned `library_id`

**Step 7.2: Create a Document & add Chunks**
- Click on **POST /libraries/{library_id}/documents**
- Replace `{library_id}` with your copied ID
- Use this JSON:
```json
{
  "document_data": {
    "metadata": {
      "title": "string",
      "author": "string",
      "created_at": "2025-09-30T06:24:03.501Z",
      "updated_at": "2025-09-30T06:24:03.501Z",
      "source": "string",
      "file_type": "string",
      "file_size": 0,
      "tags": [
        "string"
      ],
      "category": "string",
      "language": "string"
    }
  },
  "chunk_texts": [
    "Artificial intelligence (AI) is the simulation of human intelligence in machines to perform tasks and learn from experience, often involving complex reasoning, planning, and decision-making"
  ]
}
- Copy the returned `document_id`
```


```
**Step 7.3: Build Index**
- Click on **POST /libraries/{library_id}/index**
- Try different index types:
```json
{
  "index_type": "linear"
}
```

**Step 7.4: Perform Search**
- Click on **POST /libraries/{library_id}/search**
- Test semantic search:
```json
{
  "query_text": "What is artificial intelligence?",
  "top_k": 5,
  "include_text": true,
  "include_metadata": true
}
```

##  Architecture Diagram

## Complete System Architecture (Domain-Driven Design)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT APPLICATIONS                                 │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Web Browser │  │   Python SDK │  │     cURL     │  │  Mobile App  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │                  │               │
│         └──────────────────┴──────────────────┴──────────────────┘               │
│                                    │                                             │
│                              HTTP/REST API                                       │
└────────────────────────────────────┼─────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 1: API LAYER (FastAPI)                           │
│                        Handles HTTP, Validation, Routing                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐      │
│  │  library_router.py │  │ document_router.py │  │  chunk_router.py    │      │
│  ├────────────────────┤  ├────────────────────┤  ├─────────────────────┤      │
│  │ POST   /libraries  │  │ POST /libraries/   │  │ POST /libraries/    │      │
│  │ GET    /libraries  │  │      {id}/docs     │  │      {id}/chunks    │      │
│  │ GET    /libraries/ │  │ GET  /libraries/   │  │ GET  /libraries/    │      │
│  │        {id}        │  │      {id}/docs/    │  │      {id}/chunks/   │      │
│  │ PUT    /libraries/ │  │      {doc_id}      │  │      {chunk_id}     │      │
│  │        {id}        │  │ PUT  /libraries/   │  │ PUT  /libraries/    │      │
│  │ DELETE /libraries/ │  │      {id}/docs/    │  │      {id}/chunks/   │      │
│  │        {id}        │  │      {doc_id}      │  │      {chunk_id}     │      │
│  └────────────────────┘  │ DELETE /libraries/ │  │ DELETE /libraries/  │      │
│                          │      {id}/docs/    │  │      {id}/chunks/   │      │
│  ┌────────────────────┐  │      {doc_id}      │  │      {chunk_id}     │      │
│  │  search_router.py  │  └────────────────────┘  └─────────────────────┘      │
│  ├────────────────────┤                                                         │
│  │ POST /libraries/   │  ┌────────────────────┐                                │
│  │      {id}/search   │  │  index_router.py   │                                │
│  │                    │  ├────────────────────┤                                │
│  │ - Query by text    │  │ POST /libraries/   │                                │
│  │ - Query by vector  │  │      {id}/index    │                                │
│  │ - Metadata filters │  │ GET  /libraries/   │                                │
│  │ - Top K results    │  │      {id}/index/   │                                │
│  └────────────────────┘  │      stats         │                                │
│                          └────────────────────┘                                │
│                                                                                  │
│  Responsibilities:                                                            │
│  • HTTP request/response handling                                               │
│  • Pydantic model validation                                                    │
│  • Status code management                                                       │
│  • Error transformation (exceptions → HTTP errors)                              │
│  • OpenAPI/Swagger documentation generation                                     │
│  • NO business logic here!                                                      │
│                                                                                  │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   │ Calls
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: SERVICE LAYER (Business Logic)                  │
│                      Orchestrates Operations, Enforces Rules                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐   │
│  │  library_service.py │  │  document_service.py │  │  chunk_service.py   │   │
│  ├─────────────────────┤  ├──────────────────────┤  ├─────────────────────┤   │
│  │ create_library()    │  │ create_document()    │  │ create_chunk()      │   │
│  │ get_library()       │  │ get_document()       │  │ get_chunk()         │   │
│  │ update_library()    │  │ update_document()    │  │ update_chunk()      │   │
│  │ delete_library()    │  │ delete_document()    │  │ delete_chunk()      │   │
│  │ list_libraries()    │  │ add_chunks_to_doc()  │  │ bulk_create()       │   │
│  └─────────────────────┘  └──────────────────────┘  └─────────────────────┘   │
│                                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐   │
│  │  search_service.py  │  │   index_service.py   │  │ embedding_service.py│   │
│  ├─────────────────────┤  ├──────────────────────┤  ├─────────────────────┤   │
│  │ search_library()    │  │ build_index()        │  │ get_embedding()     │   │
│  │ - Apply filters     │  │ delete_index()       │  │ batch_embed()       │   │
│  │ - Call index search │  │ update_index()       │  │ - Cohere API        │   │
│  │ - Rank results      │  │ get_index_stats()    │  │ - Retry logic       │   │
│  │ - Format output     │  │ recommend_algo()     │  │ - Error handling    │   │
│  └─────────────────────┘  └──────────────────────┘  └─────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │          metadata_filter_service.py                                   │      │
│  ├──────────────────────────────────────────────────────────────────────┤      │
│  │ apply_filters() - Pre/Post filtering logic                            │      │
│  │ parse_filter_expression() - Query language parsing                    │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  Responsibilities:                                                            │
│  • Business logic and rules enforcement                                         │
│  • Transaction coordination across repositories                                 │
│  • Integration with external services (embeddings)                              │
│  • Concurrency control (async/await, locking)                                   │
│  • Algorithm selection and optimization                                         │
│  • Error handling and domain exceptions                                         │
│                                                                                  │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   │ Uses
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: REPOSITORY LAYER (Data Access)                      │
│                         Abstracts Data Storage/Retrieval                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │library_repository.py │  │document_repository.py│  │ chunk_repository.py  │ │
│  ├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤ │
│  │ create(library)      │  │ create(document)     │  │ create(chunk)        │ │
│  │ get(id) → Library    │  │ get(id) → Document   │  │ get(id) → Chunk      │ │
│  │ update(id, library)  │  │ update(id, document) │  │ update(id, chunk)    │ │
│  │ delete(id) → bool    │  │ delete(id) → bool    │  │ delete(id) → bool    │ │
│  │ list() → List[...]   │  │ get_by_library()     │  │ get_by_library()     │ │
│  │ exists(id) → bool    │  │ count() → int        │  │ get_by_document()    │ │
│  └──────────────────────┘  └──────────────────────┘  │ get_vectors()        │ │
│                                                       └──────────────────────┘ │
│                          ┌────────────────────┐                                 │
│                          │   base.py          │                                 │
│                          ├────────────────────┤                                 │
│                          │ BaseRepository     │                                 │
│                          │ - CRUD interface   │                                 │
│                          │ - Common methods   │                                 │
│                          └────────────────────┘                                 │
│                                                                                  │
│  💾 Current Implementation: In-Memory Storage                                   │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │  libraries: Dict[UUID, Library] = {}                                │        │
│  │  documents: Dict[UUID, Document] = {}                               │        │
│  │  chunks: Dict[UUID, Chunk] = {}                                     │        │
│  │                                                                      │        │
│  │  + asyncio.Lock() for thread-safety                                 │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
│  📋 Responsibilities:                                                            │
│  • CRUD operations on domain entities                                           │
│  • Data persistence abstraction                                                 │
│  • Query implementation                                                         │
│  • Thread-safe access to shared state                                           │
│  • Can be swapped: InMemory → PostgreSQL → Redis (transparent to services)     │
│                                                                                  │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │
                                   │ Stores/Retrieves
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 4: DOMAIN LAYER (Core Models)                        │
│                        Pydantic Models, Business Entities                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐       │
│  │    library.py      │  │   document.py      │  │     chunk.py       │       │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤       │
│  │ class Library:     │  │ class Document:    │  │ class Chunk:       │       │
│  │   id: UUID         │  │   id: UUID         │  │   id: UUID         │       │
│  │   name: str        │  │   library_id: UUID │  │   text: str        │       │
│  │   description: str │  │   document_id: UUID│  │   embedding: List  │       │
│  │   index_type: str  │  │   metadata: dict   │  │   document_id: UUID│       │
│  │   index_status: str│  │   chunk_count: int │  │   library_id: UUID │       │
│  │   metadata: dict   │  │   created_at: dt   │  │   metadata: dict   │       │
│  │   created_at: dt   │  │   updated_at: dt   │  │   created_at: dt   │       │
│  │   updated_at: dt   │  └────────────────────┘  └────────────────────┘       │
│  └────────────────────┘                                                         │
│                                                                                  │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐       │
│  │    search.py       │  │ metadata_filter.py │  │   pagination.py    │       │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤       │
│  │ SearchRequest      │  │ MetadataFilter     │  │ PaginationParams   │       │
│  │ SearchResponse     │  │ FilterOperator     │  │ PaginatedResponse  │       │
│  │ SearchResult       │  │ FilterExpression   │  └────────────────────┘       │
│  └────────────────────┘  └────────────────────┘                                │
│                                                                                  │
│   Responsibilities:                                                            │
│  • Define domain entities with validation                                       │
│  • Type safety with Pydantic                                                    │
│  • Automatic serialization/deserialization                                      │
│  • Field constraints and business rules                                         │
│  • No dependencies on other layers (pure domain)                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                   CROSS-CUTTING CONCERNS: INDEXING ALGORITHMS                   │
│                          (Used by Index Service)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │                        base.py (Abstract Interface)                   │      │
│  │ ┌──────────────────────────────────────────────────────────────────┐ │      │
│  │ │  class BaseIndex(ABC):                                            │ │      │
│  │ │    @abstractmethod                                                │ │      │
│  │ │    def build(vectors: List[Tuple[UUID, List[float]]]) -> None    │ │      │
│  │ │    @abstractmethod                                                │ │      │
│  │ │    def search(query: List[float], k: int) -> List[SearchResult]  │ │      │
│  │ │    @abstractmethod                                                │ │      │
│  │ │    def add(chunk_id: UUID, vector: List[float]) -> None          │ │      │
│  │ │    @abstractmethod                                                │ │      │
│  │ │    def remove(chunk_id: UUID) -> bool                             │ │      │
│  │ └──────────────────────────────────────────────────────────────────┘ │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                    │                                             │
│                                    │ implements                                  │
│                    ┌───────────────┴───────────────┐                            │
│                    ▼               ▼               ▼                            │
│  ┌───────────────────────┐  ┌──────────────┐  ┌──────────────────────┐        │
│  │  linear_index.py      │  │kd_tree_index │  │   lsh_index.py       │        │
│  ├───────────────────────┤  ├──────────────┤  ├──────────────────────┤        │
│  │ LinearIndex           │  │ KDTreeIndex  │  │ LSHIndex             │        │
│  │                       │  │              │  │                      │        │
│  │ Time: O(N×D)          │  │ Time: O(logN)│  │ Time: O(N^ρ), ρ<1   │        │
│  │ Space: O(N×D)         │  │      to O(N) │  │ Space: O(L×N×D)      │        │
│  │ Recall: 100%          │  │ Space: O(N×D)│  │ Recall: 90-95%       │        │
│  │                       │  │ Recall: 100% │  │                      │        │
│  │ ✓ Exact search        │  │              │  │ ✓ High dimensions    │        │
│  │ ✓ Simple              │  │ ✓ Fast in low│  │ ✓ Sub-linear search  │        │
│  │ ✗ Poor scalability    │  │   dimensions │  │ ✓ Tunable precision  │        │
│  │                       │  │ ✗ Curse of   │  │ ✗ Approximate        │        │
│  │ Use: <10K vectors     │  │   dimensionlty│  │                      │        │
│  └───────────────────────┘  └──────────────┘  │ Use: 100K+ vectors,  │        │
│                                                │      high-D          │        │
│  ┌──────────────────────┐  ┌───────────────┐  └──────────────────────┘        │
│  │   hnsw_index.py      │  │optimized_     │                                   │
│  ├──────────────────────┤  │linear_index.py│  ┌───────────────────────┐       │
│  │ HNSWIndex            │  ├───────────────┤  │ multiprobe_lsh_       │       │
│  │                      │  │ Optimized     │  │ index.py              │       │
│  │ Time: O(log N)       │  │ Linear        │  ├───────────────────────┤       │
│  │ Space: O(M×N×D)      │  │               │  │ MultiProbe LSH        │       │
│  │ Recall: 95-99%       │  │ + SIMD        │  │                       │       │
│  │                      │  │ + Batching    │  │ + Multi-bucket probe  │       │
│  │ ✓ State-of-the-art   │  │ + NumPy opts  │  │ + Better recall       │       │
│  │ ✓ Best recall/speed  │  └───────────────┘  └───────────────────────┘       │
│  │ ✓ Dynamic inserts    │                                                       │
│  │ ✓ Runtime tunable    │  ┌─────────────────────────────────────────┐        │
│  │ ✗ Complex impl       │  │    index_factory.py                      │        │
│  │ ✗ Higher memory      │  ├─────────────────────────────────────────┤        │
│  │                      │  │ create(type, dimension) → BaseIndex      │        │
│  │ Use: Production!     │  │ recommend_index_type() → IndexType       │        │
│  └──────────────────────┘  └─────────────────────────────────────────┘        │
│                                                                                  │
│   Design Pattern: Strategy Pattern                                            │
│  • Index Service uses BaseIndex interface                                       │
│  • Concrete implementations are interchangeable                                 │
│  • Open/Closed Principle: add new algorithms without modifying existing code    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-CUTTING CONCERNS: INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │  persistence/persistence_manager.py                                   │      │
│  ├──────────────────────────────────────────────────────────────────────┤      │
│  │  • save_state() - Snapshot to disk                                    │      │
│  │  • load_state() - Restore from disk                                   │      │
│  │  • write_wal_entry() - Write-Ahead Log                                │      │
│  │  • Periodic snapshots (every 5 min)                                   │      │
│  │  • JSON for metadata, NumPy compressed for vectors                    │      │
│  │  • Atomic writes with temp files                                      │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │  distributed/distributed_architecture.py                              │      │
│  ├──────────────────────────────────────────────────────────────────────┤      │
│  │  • Leader-Follower pattern                                            │      │
│  │  • Leader election (Raft-like consensus)                              │      │
│  │  • Heartbeat mechanism                                                │      │
│  │  • Async/Sync replication modes                                       │      │
│  │  • Reads: Any node | Writes: Leader only                              │      │
│  │  • Automatic failover on leader failure                               │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │  core/error_handlers.py                                               │      │
│  ├──────────────────────────────────────────────────────────────────────┤      │
│  │  • Custom domain exceptions                                           │      │
│  │  • HTTP exception mapping                                             │      │
│  │  • Centralized error handling                                         │      │
│  │  • Structured error responses                                         │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW EXAMPLE                                   │
│                   (Search Request: "Find similar documents")                    │
└─────────────────────────────────────────────────────────────────────────────────┘

   1. CLIENT
      │
      │ POST /libraries/{id}/search
      │ { "query_text": "machine learning", "top_k": 5 }
      ▼

   2. API LAYER (search_router.py)
      │
      │ • Validate request with Pydantic
      │ • Extract library_id, query_text, top_k
      ▼
      │ search_service.search_library(lib_id, query_text, top_k)
      ▼

   3. SERVICE LAYER (search_service.py)
      │
      │ A) Get library from repository
      │    library_repo.get(library_id) → Library
      │
      │ B) Generate embedding for query text
      │    embedding_service.get_embedding("machine learning")
      │    └─→ Call Cohere API → [0.123, -0.456, 0.789, ...]
      │
      │ C) Get index for library
      │    index_service.get_index(library_id) → HNSWIndex
      │
      │ D) Search index
      │    index.search(query_vector, k=5)
      │    └─→ HNSW traversal → [chunk1, chunk2, chunk3, chunk4, chunk5]
      │
      │ E) Apply metadata filters (if any)
      │    metadata_filter_service.apply_filters(results, filters)
      │
      │ F) Fetch full chunk data
      │    chunk_repo.get_batch([chunk_ids])
      │
      │ G) Format response
      │    SearchResponse(results=[...], total=5, search_time_ms=12.3)
      ▼

   4. API LAYER
      │
      │ Convert to JSON
      │ Return HTTP 200 with results
      ▼

   5. CLIENT
      │
      │ Receives search results with similarity scores
      └─→ Display to user


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KEY DESIGN PRINCIPLES                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SOLID PRINCIPLES                                                             │
│                                                                                  │
│  S - Single Responsibility                                                       │
│      Each class has one reason to change                                        │
│      Example: LinearIndex only handles linear search                            │
│                                                                                  │
│  O - Open/Closed                                                                 │
│      Open for extension, closed for modification                                │
│      Example: Add new index types without changing search service               │
│                                                                                  │
│  L - Liskov Substitution                                                         │
│      Subclasses can replace parent classes                                      │
│      Example: Any BaseIndex implementation works in search service              │
│                                                                                  │
│  I - Interface Segregation                                                       │
│      Clients don't depend on unused methods                                     │
│      Example: Search doesn't need to know about index building internals        │
│                                                                                  │
│  D - Dependency Inversion                                                        │
│      Depend on abstractions, not concretions                                    │
│      Example: Service → BaseIndex interface, not specific index class           │
│                                                                                  │
│  ────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│   DOMAIN-DRIVEN DESIGN                                                        │
│                                                                                  │
│  • Layered Architecture - Clear separation of concerns                          │
│  • Repository Pattern - Data access abstraction                                 │
│  • Service Layer - Business logic isolation                                     │
│  • Domain Models - Rich, validated entities (Pydantic)                          │
│  • Ubiquitous Language - Consistent terminology (Library, Chunk, Index)         │
│                                                                                  │
│  ────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│   CONCURRENCY & THREAD SAFETY                                                  │
│                                                                                  │
│  • async/await throughout (non-blocking I/O)                                    │
│  • asyncio.Lock for shared state access                                         │
│  • Thread-safe repository operations                                            │
│  • FastAPI handles concurrent requests automatically                            │
│                                                                                  │
│  ────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│   BEST PRACTICES                                                               │
│                                                                                  │
│  • Type hints everywhere (static typing)                                        │
│  • Pydantic for validation (fail fast)                                          │
│  • Early returns (avoid deep nesting)                                           │
│  • Composition over inheritance                                                 │
│  • No hardcoded values (use constants)                                          │
│  • Comprehensive error handling                                                 │
│  • Unit + Integration testing                                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

------------------------------ TO DO
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────┐
                        │      Load Balancer      │
                        └────────────┬────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
         ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
         │  Docker       │   │  Docker       │   │  Docker       │
         │  Container 1  │   │  Container 2  │   │  Container 3  │
         │               │   │               │   │               │
         │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────┐ │
         │ │ FastAPI   │ │   │ │ FastAPI   │ │   │ │ FastAPI   │ │
         │ │ App       │ │   │ │ App       │ │   │ │ App       │ │
         │ └───────────┘ │   │ └───────────┘ │   │ └───────────┘ │
         │               │   │               │   │               │
         │  Leader       │   │  Follower     │   │  Follower     │
         │  (Writes)     │   │  (Reads)      │   │  (Reads)      │
         └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                 │                   │                   │
                 └───────────────────┼───────────────────┘
                                     │
                          Replication & Sync
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   Persistent Storage  │
                          │                      │
                          │  data/               │
                          │  ├─ snapshots/       │
                          │  ├─ wal/             │
                          │  └─ indexes/         │
                          └──────────────────────┘

         Docker Volume: vectordb_data (mounted to /app/data)
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
-->

###  Test CRUD Operations

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

###  Performance Testing

1. Create a library with many chunks (50+)
2. Compare search performance:
   - Without index (linear scan)
   - With KD-Tree index
   - With LSH index
3. Monitor response times in the Swagger UI

### Expected Results

 **Successful Library Creation**: Returns library with unique ID
 **Chunk Creation**: Automatically generates 1024-dimensional embeddings
 **Semantic Search**: Returns relevant chunks even with different wording
 **Similarity Scores**: Range from 0.0 to 1.0 (higher is more similar)
 **Index Building**: Improves search performance for large datasets
 **Metadata Filtering**: Correctly filters results based on criteria

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

## Running Automated Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api_integration.py -v
```




