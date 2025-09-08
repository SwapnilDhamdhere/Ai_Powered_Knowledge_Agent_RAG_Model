from PyPDF2 import PdfReader
from app.core.logger import logger

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        pdf = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise RuntimeError(f"PDF parsing failed: {e}")
