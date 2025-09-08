from pydantic import BaseModel
from typing import List

class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload API.
    """
    message: str
    chunks: int
    source: str
