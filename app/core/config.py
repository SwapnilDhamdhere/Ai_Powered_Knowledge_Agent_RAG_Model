import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    """Centralized app configuration."""

    # App metadata
    APP_NAME: str = os.getenv("APP_NAME", "AI Knowledge Agent")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    APP_DESC: str = os.getenv(
        "APP_DESCRIPTION",
        "AI-powered semantic search + RAG with GPT-OSS (Ollama) + Qdrant, exposed via FastAPI."
    )

    # Ollama Config
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_EMBEDDINGS_URL: str = f"{OLLAMA_HOST}/api/embed"
    OLLAMA_CHAT_URL: str = f"{OLLAMA_HOST}/api/chat"
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    OLLAMA_EMBEDDINGS_MODEL: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

    # Qdrant Config
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "knowledge_base")
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", 768))
    QDRANT_DISTANCE: str = os.getenv("QDRANT_DISTANCE", "COSINE")

    # HNSW Parameters
    QDRANT_HNSW_M: int = int(os.getenv("QDRANT_HNSW_M", 32))
    QDRANT_HNSW_EF_CONSTRUCT: int = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", 128))
    QDRANT_FULL_SCAN_THRESHOLD: int = int(
        os.getenv("QDRANT_FULL_SCAN_THRESHOLD", 0)
    ) or (10 * int(os.getenv("QDRANT_VECTOR_SIZE", 768)))  # ✅ Auto-tune fallback

    # Document Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))

    # Debug & Logging
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Search tuning
    TOP_K: int = int(os.getenv("TOP_K", 8))
    MIN_CHUNKS: int = int(os.getenv("MIN_CHUNKS", 3))
    MIN_RELEVANCE: float = float(os.getenv("MIN_RELEVANCE", 0.6))

    # Batching
    EMBEDDINGS_BATCH_SIZE: int = int(os.getenv("EMBEDDINGS_BATCH_SIZE", 64))
    QDRANT_UPSERT_BATCH_SIZE: int = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", 128))

    # Search mode: "semantic" or "hybrid"
    SEARCH_MODE: str = os.getenv("SEARCH_MODE", "hybrid").lower()


settings = Settings()
