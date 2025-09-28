"""
FastAPI Vector Database Application.

Main application entry point with all routers and middleware.
Following Clean Code principles: meaningful names, single responsibility, no hardcoding.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.library_router import router as library_router
from api.document_router import router as document_router
from api.chunk_router import router as chunk_router
from api.search_router import router as search_router
from core.constants import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    API_MESSAGE_ROOT,
    API_STATUS_HEALTHY,
    API_STATUS_RUNNING,
    CONTACT_NAME,
    CONTACT_URL,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_STATE_FILE_PATH,
    ENDPOINT_ROOT,
    ENDPOINT_HEALTH,
    ENDPOINT_DOCS,
    ENV_HOST,
    ENV_PORT,
    FEATURE_VECTOR_SEARCH,
    FEATURE_MULTIPLE_INDEXES,
    FEATURE_METADATA_FILTERING,
    FEATURE_BATCH_OPERATIONS,
    STARTUP_MESSAGE,
    SHUTDOWN_MESSAGE,
    LOADING_DATA_MESSAGE,
    SAVING_DATA_MESSAGE,
    SERVER_START_MESSAGE,
    CORS_ALLOW_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS,
)


def print_startup_message() -> None:
    """Print application startup message."""
    print(STARTUP_MESSAGE)


def print_shutdown_message() -> None:
    """Print application shutdown message."""
    print(SHUTDOWN_MESSAGE)


def should_load_persisted_data() -> bool:
    """Check if persisted data should be loaded."""
    return os.path.exists(DEFAULT_STATE_FILE_PATH)


def load_persisted_data_if_available() -> None:
    """Load persisted data if file exists."""
    if should_load_persisted_data():
        print(LOADING_DATA_MESSAGE)
        # TODO: Implement actual data loading
        # load_state(DEFAULT_STATE_FILE_PATH)


def save_application_state() -> None:
    """Save current application state to disk."""
    print(SAVING_DATA_MESSAGE)
    # TODO: Implement actual state saving
    # save_state(DEFAULT_STATE_FILE_PATH)


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """
    Application lifespan manager following single responsibility principle.

    Handles startup and shutdown events cleanly.
    """
    # Startup phase
    print_startup_message()
    load_persisted_data_if_available()

    yield

    # Shutdown phase
    save_application_state()
    print_shutdown_message()


def create_fastapi_application() -> FastAPI:
    """
    Create and configure FastAPI application.

    Separates application creation from configuration for better testability.
    """
    return FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        contact={
            "name": CONTACT_NAME,
            "url": CONTACT_URL,
        },
        lifespan=application_lifespan,
    )


def configure_cors_middleware(application: FastAPI) -> None:
    """Configure CORS middleware with constants."""
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,  # TODO: Restrict in production
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )


def register_api_routers(application: FastAPI) -> None:
    """Register all API routers with the application."""
    application.include_router(library_router)
    application.include_router(document_router)
    application.include_router(chunk_router)
    application.include_router(search_router)

    # Temporary debug routes
    from debug_endpoint import add_debug_routes
    add_debug_routes(application)


def create_health_check_response() -> dict:
    """Create health check response with embedding service status."""
    from api.dependencies import get_embedding_service

    embedding_service = get_embedding_service()

    return {
        "status": API_STATUS_HEALTHY,
        "embedding_service": {
            "provider": embedding_service.provider,
            "available": embedding_service.is_available(),
            "dimension": embedding_service.dimension,
        },
        "features": {
            "vector_search": FEATURE_VECTOR_SEARCH,
            "multiple_indexes": FEATURE_MULTIPLE_INDEXES,
            "metadata_filtering": FEATURE_METADATA_FILTERING,
            "batch_operations": FEATURE_BATCH_OPERATIONS,
        }
    }


def create_root_response() -> dict:
    """Create root endpoint response with API information."""
    return {
        "message": API_MESSAGE_ROOT,
        "version": API_VERSION,
        "docs": ENDPOINT_DOCS,
        "health": ENDPOINT_HEALTH,
        "status": API_STATUS_RUNNING
    }


def get_server_configuration() -> tuple[str, int]:
    """Get server host and port from environment variables."""
    host = os.getenv(ENV_HOST, DEFAULT_HOST)
    port = int(os.getenv(ENV_PORT, DEFAULT_PORT))
    return host, port


def print_server_start_message(host: str, port: int) -> None:
    """Print server startup message with host and port."""
    print(f"{SERVER_START_MESSAGE} on {host}:{port}")


def start_development_server() -> None:
    """Start development server with configuration from environment."""
    import uvicorn

    host, port = get_server_configuration()
    print_server_start_message(host, port)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload in development
        log_level=DEFAULT_LOG_LEVEL,
    )


# Create FastAPI application using factory functions
app = create_fastapi_application()
configure_cors_middleware(app)
register_api_routers(app)


@app.get(ENDPOINT_ROOT, summary="Root endpoint")
async def root_endpoint():
    """Root endpoint providing basic API information."""
    return create_root_response()


@app.get(ENDPOINT_HEALTH, summary="Health check")
async def health_check_endpoint():
    """Health check endpoint for monitoring."""
    return create_health_check_response()


if __name__ == "__main__":
    start_development_server()