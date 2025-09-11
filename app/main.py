import os
import time
import uuid
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app import __app_name__, __version__, logger
from app.routes import api_router
from app.core.config import settings

# We try to import async ensure_collection from qdrant service (preferred).
# If unavailable, we'll fallback to the old synchronous initializer if present.
try:
    # preferred async ensure_collection which we added in qdrant_service
    from app.services.qdrant_service import ensure_collection as ensure_collection_async  # type: ignore
    ENSURE_COLLECTION_ASYNC = True
except Exception:
    ensure_collection_async = None
    ENSURE_COLLECTION_ASYNC = False

# Fallback: older qdrant init (sync) - kept for backward compatibility
try:
    from app.db.qdrant_init import init_qdrant, ensure_collection as ensure_collection_sync  # type: ignore
    HAS_SYNC_QDRANT_INIT = True
except Exception:
    init_qdrant = None
    ensure_collection_sync = None
    HAS_SYNC_QDRANT_INIT = False

# Ollama health check (async) if available
try:
    from app.services.ollama_service import ollama_health_check  # type: ignore
    HAS_OLLAMA_HEALTH = True
except Exception:
    ollama_health_check = None
    HAS_OLLAMA_HEALTH = False

# -------------------------------------------------------------------
# App metadata / description (can be moved to config if desired)
# -------------------------------------------------------------------
APP_TITLE = os.getenv("APP_TITLE", "AI-Powered Knowledge Agent")
APP_DESC = os.getenv(
    "APP_DESCRIPTION",
    "Semantic search + RAG with Ollama (GPT-OSS) and Qdrant, exposed via FastAPI."
)

# -------------------------------------------------------------------
# Lifespan: startup & shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown hooks.
    - Ensure Qdrant collection exists (async preferred).
    - Call legacy init if available.
    """
    # Startup
    logger.info(json.dumps({"event": "startup", "app": __app_name__, "version": __version__}))

    # Preferred path: async ensure collection in qdrant_service
    if ENSURE_COLLECTION_ASYNC and ensure_collection_async is not None:
        try:
            await ensure_collection_async()
            logger.info(json.dumps({"event": "qdrant", "status": "collection_ensured_async"}))
        except Exception as e:
            logger.error(json.dumps({"event": "qdrant", "error": str(e)}))
    else:
        # Fallback: legacy sync init (best-effort - do not raise if missing)
        if HAS_SYNC_QDRANT_INIT and init_qdrant is not None:
            try:
                init_qdrant()
                logger.info(json.dumps({"event": "qdrant", "status": "initialized_sync"}))
            except Exception as e:
                logger.error(json.dumps({"event": "qdrant", "error": str(e)}))

    yield

    # Shutdown
    logger.info(json.dumps({"event": "shutdown", "app": __app_name__}))


# -------------------------------------------------------------------
# FastAPI app instance
# -------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    version=__version__,
    description=APP_DESC,
    lifespan=lifespan,
)

# -------------------------------------------------------------------
# CORS (open for development; restrict in production)
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to trusted origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Request timing + request id middleware
# -------------------------------------------------------------------
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """
    Attach a request id, record processing duration, and log a structured event.
    Adds headers:
      - X-Request-ID
      - X-Process-Time
    """
    request_id = str(uuid.uuid4())
    start = time.time()

    # Attach request_id to request.state so downstream code can use it if needed
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception as exc:
        # If an unhandled exception happens, still measure time and log it, then re-raise
        duration = time.time() - start
        # Log structured error
        logger.error(json.dumps({
            "event": "request_error",
            "method": request.method,
            "path": request.url.path,
            "request_id": request_id,
            "duration": duration,
            "error": str(exc)
        }))
        raise

    duration = time.time() - start
    # Add headers for tracing
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}"

    # Structured log for request
    logger.info(json.dumps({
        "event": "request",
        "method": request.method,
        "path": request.url.path,
        "request_id": request_id,
        "duration": duration,
        "status_code": response.status_code
    }))

    return response

# -------------------------------------------------------------------
# Include primary router
# -------------------------------------------------------------------
app.include_router(api_router)

# -------------------------------------------------------------------
# Health & readiness endpoints
# -------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "app": __app_name__,
        "version": __version__,
        "status": "ok",
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs",
        "redoc": "/redoc",
    }

@app.get("/healthz", tags=["Health"])
def healthz():
    return {"status": "ok"}

@app.get("/readyz", tags=["Health"])
async def readyz():
    """
    Readiness: checks Qdrant collection presence and Ollama health (if available).
    Returns JSON:
      {
        "ready": bool,
        "qdrant_ok": bool,
        "ollama_ok": bool
      }
    """
    qdrant_ok = False
    ollama_ok = False

    # Check Qdrant: try the async ensure_collection or the legacy client
    try:
        if ENSURE_COLLECTION_ASYNC and ensure_collection_async is not None:
            # If ensure_collection runs without exception, assume OK
            await ensure_collection_async()
            qdrant_ok = True
        elif HAS_SYNC_QDRANT_INIT and ensure_collection_sync is not None:
            try:
                ensure_collection_sync()
                qdrant_ok = True
            except Exception:
                qdrant_ok = False
        else:
            qdrant_ok = False
    except Exception as e:
        qdrant_ok = False
        logger.error(json.dumps({"event": "readyz_qdrant_error", "error": str(e)}))

    # Check Ollama health if service present
    if HAS_OLLAMA_HEALTH and ollama_health_check is not None:
        try:
            ollama_ok = await ollama_health_check()
        except Exception as e:
            ollama_ok = False
            logger.error(json.dumps({"event": "readyz_ollama_error", "error": str(e)}))

    ready = qdrant_ok and (ollama_ok if HAS_OLLAMA_HEALTH else True)
    return {"ready": ready, "qdrant_ok": qdrant_ok, "ollama_ok": ollama_ok}

# -------------------------------------------------------------------
# Local dev entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )