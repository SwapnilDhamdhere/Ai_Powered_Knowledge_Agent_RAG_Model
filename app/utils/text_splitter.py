import re

def split_text(text: str, max_tokens: int = 512):
    """
    Split text into smaller chunks without breaking sentences.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
