from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logger import logger

class QdrantConnectionError(Exception):
    """Raised when Qdrant cannot be reached or queried."""
    pass

class OllamaConnectionError(Exception):
    """Raised when Ollama API is not responding."""
    pass

async def generic_exception_handler(request: Request, exc: Exception):
    """Handles unexpected server errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)},
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles validation errors from FastAPI."""
    logger.warning(f"Validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()},
    )
