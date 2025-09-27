"""
Common error handling utilities following Clean Code principles.

Provides reusable error handling functions to avoid code duplication (DRY principle).
Each function has a single responsibility and meaningful names.
"""

from uuid import UUID
from fastapi import HTTPException, status
from repository.base import NotFoundError, DuplicateError
from core.constants import (
    ERROR_LIBRARY_NOT_FOUND,
    ERROR_DOCUMENT_NOT_FOUND,
    ERROR_CHUNK_NOT_FOUND,
)


def create_not_found_exception(resource_type: str, resource_id: UUID) -> HTTPException:
    """
    Create HTTP 404 exception for any resource not found.

    Args:
        resource_type: Type of resource (library, document, chunk)
        resource_id: ID of the resource that was not found

    Returns:
        HTTPException with 404 status code
    """
    error_messages = {
        "library": ERROR_LIBRARY_NOT_FOUND,
        "document": ERROR_DOCUMENT_NOT_FOUND,
        "chunk": ERROR_CHUNK_NOT_FOUND,
    }

    error_message = error_messages.get(resource_type, f"{resource_type.title()} not found")

    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{error_message}: {resource_id}"
    )


def create_duplicate_resource_exception(error: DuplicateError) -> HTTPException:
    """
    Create HTTP 400 exception for duplicate resource creation.

    Args:
        error: The duplicate error from repository layer

    Returns:
        HTTPException with 400 status code
    """
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error)
    )


def create_internal_server_error_exception(operation: str, error: Exception) -> HTTPException:
    """
    Create HTTP 500 exception for unexpected internal errors.

    Args:
        operation: Description of the operation that failed
        error: The unexpected error that occurred

    Returns:
        HTTPException with 500 status code
    """
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to {operation}: {str(error)}"
    )


def create_validation_error_exception(field_name: str, validation_message: str) -> HTTPException:
    """
    Create HTTP 400 exception for validation errors.

    Args:
        field_name: Name of the field that failed validation
        validation_message: Description of the validation failure

    Returns:
        HTTPException with 400 status code
    """
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Validation error for {field_name}: {validation_message}"
    )


def handle_service_exceptions(operation: str, resource_type: str = "resource"):
    """
    Decorator to handle common service exceptions with clean error responses.

    Args:
        operation: Description of the operation being performed
        resource_type: Type of resource for not found errors

    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except NotFoundError as not_found_error:
                # Extract resource ID from error if available
                resource_id = getattr(not_found_error, 'resource_id', 'unknown')
                raise create_not_found_exception(resource_type, resource_id)
            except DuplicateError as duplicate_error:
                raise create_duplicate_resource_exception(duplicate_error)
            except ValueError as validation_error:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(validation_error)
                )
            except Exception as unexpected_error:
                raise create_internal_server_error_exception(operation, unexpected_error)

        return wrapper
    return decorator


def validate_positive_integer(value: int, field_name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Raises:
        HTTPException: If value is not positive
    """
    if value <= 0:
        raise create_validation_error_exception(
            field_name,
            "must be a positive integer"
        )


def validate_non_negative_integer(value: int, field_name: str) -> None:
    """
    Validate that a value is a non-negative integer.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages

    Raises:
        HTTPException: If value is negative
    """
    if value < 0:
        raise create_validation_error_exception(
            field_name,
            "must be non-negative"
        )


def validate_string_length(value: str, field_name: str, min_length: int = 1, max_length: int = 255) -> None:
    """
    Validate string length constraints.

    Args:
        value: String to validate
        field_name: Name of the field for error messages
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Raises:
        HTTPException: If string length is invalid
    """
    if len(value) < min_length:
        raise create_validation_error_exception(
            field_name,
            f"must be at least {min_length} characters long"
        )

    if len(value) > max_length:
        raise create_validation_error_exception(
            field_name,
            f"must be no more than {max_length} characters long"
        )