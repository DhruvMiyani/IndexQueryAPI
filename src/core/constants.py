"""
Application constants to avoid hardcoded values.

Following Clean Code principle: "Stop Hardcoding Values"
"""

# API Configuration
API_TITLE = "Vector Database API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
A Vector Database REST API for indexing and querying document embeddings.

## Features

* **Libraries**: Create and manage collections of documents
* **Documents**: Organize chunks of text with metadata
* **Chunks**: Text pieces with vector embeddings
* **Vector Search**: k-nearest neighbor similarity search
* **Multiple Indexes**: Linear, KD-Tree, and LSH algorithms
* **Metadata Filtering**: Filter search results by metadata

## Usage

1. Create a library to organize your documents
2. Add documents with chunks of text
3. Build a vector index for fast search
4. Perform similarity searches using text queries or vectors

The API automatically handles embedding generation using Cohere API.
"""

# Server Configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"

# Pagination Configuration
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000
DEFAULT_PAGE_OFFSET = 0

# API Messages
API_STATUS_RUNNING = "running"
API_STATUS_HEALTHY = "healthy"
API_MESSAGE_ROOT = "Vector Database API"

# Health Check Feature Flags
FEATURE_VECTOR_SEARCH = True
FEATURE_MULTIPLE_INDEXES = True
FEATURE_METADATA_FILTERING = True
FEATURE_BATCH_OPERATIONS = True

# Embedding Service Configuration
EMBEDDING_PROVIDER_COHERE = "cohere"
EMBEDDING_DIMENSION_DEFAULT = 1024

# Index Status Constants
INDEX_STATUS_NONE = "none"
INDEX_STATUS_BUILDING = "building"
INDEX_STATUS_READY = "ready"
INDEX_STATUS_ERROR = "error"

# Index Types
INDEX_TYPE_LINEAR = "linear"
INDEX_TYPE_KD_TREE = "kd_tree"
INDEX_TYPE_LSH = "lsh"

# Default Search Configuration
DEFAULT_SEARCH_TOP_K = 10
MAX_SEARCH_TOP_K = 100
MINIMUM_SEARCH_TOP_K = 1

# Error Messages
ERROR_LIBRARY_NOT_FOUND = "Library not found"
ERROR_DOCUMENT_NOT_FOUND = "Document not found"
ERROR_CHUNK_NOT_FOUND = "Chunk not found"
ERROR_LIBRARY_CREATION_FAILED = "Failed to create library"
ERROR_DOCUMENT_CREATION_FAILED = "Failed to create document"
ERROR_CHUNK_CREATION_FAILED = "Failed to create chunk"
ERROR_SEARCH_FAILED = "Search operation failed"
ERROR_INDEX_BUILD_FAILED = "Failed to build index"

# Success Messages
SUCCESS_LIBRARY_CREATED = "Library created successfully"
SUCCESS_LIBRARY_UPDATED = "Library updated successfully"
SUCCESS_LIBRARY_DELETED = "Library deleted successfully"
SUCCESS_DOCUMENT_CREATED = "Document created successfully"
SUCCESS_CHUNK_CREATED = "Chunk created successfully"

# Application Lifecycle Messages
STARTUP_MESSAGE = "🚀 Vector Database API starting up..."
SHUTDOWN_MESSAGE = "💤 Vector Database API shutting down..."
SERVER_START_MESSAGE = "🌟 Starting Vector Database API"
LOADING_DATA_MESSAGE = "📂 Loading persisted data..."
SAVING_DATA_MESSAGE = "💾 Saving state to disk..."

# File Paths
DATA_DIRECTORY = "data"
STATE_FILE_NAME = "state.json"
DEFAULT_STATE_FILE_PATH = f"{DATA_DIRECTORY}/{STATE_FILE_NAME}"

# CORS Configuration (Development - restrict in production)
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# Environment Variable Names
ENV_HOST = "HOST"
ENV_PORT = "PORT"
ENV_COHERE_API_KEY = "COHERE_API_KEY"
ENV_DATA_DIR = "DATA_DIR"
ENV_LOG_LEVEL = "LOG_LEVEL"

# Contact Information
CONTACT_NAME = "Vector Database API"
CONTACT_URL = "https://github.com/your-repo/vectordb-api"

# HTTP Endpoints
ENDPOINT_ROOT = "/"
ENDPOINT_HEALTH = "/health"
ENDPOINT_DOCS = "/docs"
ENDPOINT_STATISTICS = "/statistics"

# Similarity Metrics
SIMILARITY_METRIC_COSINE = "cosine"
SIMILARITY_METRIC_EUCLIDEAN = "euclidean"
SIMILARITY_METRIC_DOT_PRODUCT = "dot_product"

# Vector Database Limits
MAX_EMBEDDING_DIMENSION = 2048
MIN_EMBEDDING_DIMENSION = 1
MAX_TEXT_LENGTH = 10000
MAX_LIBRARY_NAME_LENGTH = 255
MAX_DOCUMENT_NAME_LENGTH = 255

# Cache Configuration
DEFAULT_CACHE_SIZE = 1000
DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance Thresholds
PERFORMANCE_WARNING_THRESHOLD_MS = 1000
PERFORMANCE_SLOW_QUERY_THRESHOLD_MS = 5000

# Validation Constants
UUID_VERSION = 4
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"