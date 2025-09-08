import os
import uuid
from fastapi import UploadFile
from app.core.logger import logger

UPLOAD_DIR = os.path.join("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to disk and return file path.
    """
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise RuntimeError(f"File saving failed: {e}")

def remove_file(file_path: str):
    """
    Remove a file safely.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
