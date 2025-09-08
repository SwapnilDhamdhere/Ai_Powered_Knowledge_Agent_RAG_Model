import re
import uuid

def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing unwanted characters.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    return text.strip()

def generate_uuid() -> str:
    """
    Generate a unique identifier.
    """
    return str(uuid.uuid4())
