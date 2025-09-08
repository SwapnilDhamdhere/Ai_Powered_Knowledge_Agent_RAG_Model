# app/models/__init__.py

from app.models.document_model import DocumentUploadResponse
from app.models.query_model import AskQuery, AskResponse

__all__ = [
    "DocumentUploadResponse",
    "AskQuery",
    "AskResponse"
]
