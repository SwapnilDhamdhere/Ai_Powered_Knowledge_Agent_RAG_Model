from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.document_model import DocumentUploadResponse
from app.services.embeddings_service import generate_embeddings_batch
from app.services.qdrant_service import upsert_points, ensure_collection
from app.utils.file_handler import save_uploaded_file, remove_file
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_splitter import split_text
from app.utils.helpers import clean_text
from app.utils.semantic_chunker import semantic_chunk_text
from app.core.config import settings
from app.core.logger import logger
from qdrant_client.http.models import PointStruct
import uuid
import os

router = APIRouter()

# @router.post("/upload", response_model=DocumentUploadResponse)
# async def upload_document(file: UploadFile = File(...)):
#     file_path = None
#     try:
#         file_path = await save_uploaded_file(file)
#         logger.info(f"Uploaded file saved: {file_path}")
#
#         if file.filename.endswith(".txt"):
#             with open(file_path, "r", encoding="utf-8") as f:
#                 full_text = f.read()
#         elif file.filename.endswith(".pdf"):
#             full_text = extract_text_from_pdf(file_path)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")
#
#         full_text = clean_text(full_text)
#         # chunks = split_text(full_text, settings.CHUNK_SIZE)
#         chunks = await semantic_chunk_text(full_text, max_tokens=settings.CHUNK_SIZE)
#         logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")
#
#         # Batch embeddings
#         texts = [c for c in chunks]
#         embeddings = await generate_embeddings_batch(texts, batch_size=settings.EMBEDDINGS_BATCH_SIZE)
#
#         # Build Qdrant points
#         points = []
#         for i, vec in enumerate(embeddings):
#             payload = {"content": chunks[i], "source": file.filename, "chunk_index": i}
#             points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
#
#         # Ensure collection exists (idempotent)
#         await ensure_collection()
#
#         # Upsert in batches
#         await upsert_points(points, batch_size=settings.QDRANT_UPSERT_BATCH_SIZE)
#         logger.info(f"Stored embeddings for {file.filename} successfully")
#
#         return DocumentUploadResponse(
#             message=f"File '{file.filename}' uploaded and processed successfully.",
#             chunks=len(chunks),
#             source=file.filename
#         )
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Upload failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
#     finally:
#         if file_path and os.path.exists(file_path):
#             remove_file(file_path)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_path = None
    try:
        file_path = await save_uploaded_file(file)
        logger.info(f"Uploaded file saved: {file_path}")

        # ✅ Use your structured PDF parser when PDF is uploaded
        if file.filename.lower().endswith(".pdf"):
            from app.utils.structured_pdf_parser import structured_pdf_parser
            chunks_data = structured_pdf_parser(file_path)
            # Extract only text for embedding
            texts = [chunk["text"] for chunk in chunks_data]
        elif file.filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = clean_text(f.read())
            texts = await semantic_chunk_text(full_text, max_tokens=settings.CHUNK_SIZE)
            chunks_data = [{"text": t, "metadata": {"source": file.filename}} for t in texts]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or TXT.")

        logger.info(f"Extracted {len(chunks_data)} structured chunks from {file.filename}")

        # ✅ Generate embeddings
        embeddings = await generate_embeddings_batch(texts, batch_size=settings.EMBEDDINGS_BATCH_SIZE)

        # ✅ Build Qdrant payloads
        points = []
        for i, vec in enumerate(embeddings):
            meta = chunks_data[i].get("metadata", {})
            payload = {
                "content": chunks_data[i]["text"],
                "source": meta.get("doc_title", file.filename),
                "section_path": meta.get("section_path"),
                "chunk_index": meta.get("chunk_index", i)
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

        # ✅ Upsert into Qdrant
        await ensure_collection()
        await upsert_points(points, batch_size=settings.QDRANT_UPSERT_BATCH_SIZE)

        logger.info(f"Stored structured embeddings for {file.filename} successfully")

        return DocumentUploadResponse(
            message=f"File '{file.filename}' uploaded and processed successfully.",
            chunks=len(chunks_data),
            source=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")
    finally:
        if file_path and os.path.exists(file_path):
            remove_file(file_path)