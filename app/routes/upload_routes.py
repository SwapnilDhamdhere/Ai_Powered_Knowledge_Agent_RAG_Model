from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.document_model import DocumentUploadResponse
from app.services.embeddings_service import generate_embedding
from app.services.qdrant_service import insert_embeddings
from app.utils.file_handler import save_uploaded_file, remove_file
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_splitter import split_text
from app.utils.helpers import clean_text
from app.core.config import settings
from app.core.logger import logger
import uuid
import os

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, generate embeddings, and store them in Qdrant.
    """
    file_path = None
    try:
        # ✅ Save uploaded file
        file_path = await save_uploaded_file(file)
        logger.info(f"Uploaded file saved: {file_path}")

        # ✅ Extract text based on file type
        if file.filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        elif file.filename.endswith(".pdf"):
            full_text = extract_text_from_pdf(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")

        # ✅ Clean and split text into chunks
        full_text = clean_text(full_text)
        chunks = split_text(full_text, settings.CHUNK_SIZE)
        logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")

        # ✅ Generate embeddings & prepare payloads
        embeddings = []
        payloads = []
        for i, chunk in enumerate(chunks):
            embedding = await generate_embedding(chunk)
            embeddings.append(embedding)
            payloads.append({
                "id": str(uuid.uuid4()),
                "content": chunk,
                "source": file.filename,
                "chunk_index": i
            })

        # ✅ Store embeddings in Qdrant
        await insert_embeddings(embeddings, payloads)
        logger.info(f"Stored embeddings for {file.filename} successfully")

        # ✅ Return response
        return DocumentUploadResponse(
            message=f"File '{file.filename}' uploaded and processed successfully.",
            chunks=len(chunks),
            source=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
    finally:
        # ✅ Clean up file after processing to save disk space
        if file_path and os.path.exists(file_path):
            remove_file(file_path)
