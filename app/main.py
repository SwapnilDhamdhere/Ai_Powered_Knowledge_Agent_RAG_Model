import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import logger, __version__, __app_name__
from app.routes import api_router
from app.db.qdrant_init import init_qdrant
from app.core.config import settings

# -------------------------------------------------------------------
# App Metadata (can later be moved to app/core/config.py)
# -------------------------------------------------------------------
APP_TITLE = os.getenv("APP_TITLE", "AI-Powered Knowledge Agent")
APP_DESC = os.getenv(
    "APP_DESCRIPTION",
    "Semantic search + RAG with Ollama (GPT-OSS) and Qdrant, exposed via FastAPI."
)

# -------------------------------------------------------------------
# Optional Routers & Initializers
# -------------------------------------------------------------------
def _optional_imports():
    upload_router = ask_router = None
    ensure_collection = None

    try:
        from app.routes.upload_routes import router as _upload_router  # type: ignore
        upload_router = _upload_router
    except Exception as e:
        logger.info("upload_routes not ready yet: %s", e)

    try:
        from app.routes.ask_routes import router as _ask_router  # type: ignore
        ask_router = _ask_router
    except Exception as e:
        logger.info("ask_routes not ready yet: %s", e)

    try:
        from app.db.qdrant_init import ensure_collection as _ensure_collection  # type: ignore
        ensure_collection = _ensure_collection
    except Exception as e:
        logger.info("qdrant_init not ready yet: %s", e)

    return upload_router, ask_router, ensure_collection

# -------------------------------------------------------------------
# Lifespan Event Hook: Startup & Shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    upload_router, ask_router, ensure_collection = _optional_imports()

    # Initialize Qdrant
    try:
        init_qdrant()
        logger.info("Qdrant client initialized ✅")
    except Exception as e:
        logger.error("Failed to initialize Qdrant client: %s", e)

    # Ensure collection exists
    if callable(ensure_collection):
        try:
            ensure_collection()
            logger.info("Qdrant collection ensured ✅")
        except Exception as e:
            logger.error("Failed to ensure Qdrant collection: %s", e)

    # Attach optional routers
    if upload_router is not None:
        app.include_router(upload_router, prefix="/api/docs", tags=["Docs"])
    if ask_router is not None:
        app.include_router(ask_router, prefix="/api", tags=["Q&A"])

    logger.info("%s v%s started", __app_name__, __version__)
    yield
    logger.info("%s shutting down", __app_name__)

# -------------------------------------------------------------------
# FastAPI App Instance
# -------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    version=__version__,
    description=APP_DESC,
    lifespan=lifespan,
)

# -------------------------------------------------------------------
# CORS Configuration (Open for Now)
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Include Primary API Router
# -------------------------------------------------------------------
app.include_router(api_router)

# -------------------------------------------------------------------
# Health Endpoints
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
    """Check Ollama & Qdrant connectivity."""
    from app.db.qdrant_init import qdrant_client
    from app.services.ollama_service import ollama_health_check

    # Check Qdrant
    try:
        collections = qdrant_client.get_collections()
        qdrant_ok = settings.QDRANT_COLLECTION in [
            c.name for c in collections.collections
        ]
    except Exception as e:
        logger.error("Qdrant check failed: %s", e)
        qdrant_ok = False

    # Check Ollama
    ollama_ok = await ollama_health_check()

    return {
        "ready": qdrant_ok and ollama_ok,
        "qdrant_ok": qdrant_ok,
        "ollama_ok": ollama_ok
    }

# -------------------------------------------------------------------
# Local Development Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
