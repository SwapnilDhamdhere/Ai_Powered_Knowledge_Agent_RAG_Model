import os
import re
import uuid
import fitz  # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Optional OCR support
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class UniversalPDFParser:
    """
    A robust PDF parser that creates hierarchical, structured chunks of text.
    Uses TOC if available, else falls back to font/visual analysis + OCR for scanned PDFs.
    """

    def __init__(self, chunk_size_chars: int = 4000, ocr_language: str = 'eng'):
        self.chunk_size_chars = chunk_size_chars
        self.ocr_language = ocr_language

    def _normalize_text(self, text: str) -> str:
        """Collapse whitespace and clean up text."""
        return re.sub(r'\s+', ' ', text).strip()

    def _split_text_into_chunks(
        self, text: str, doc_title: str, doc_path: str, section_path: List[str], start_index: int
    ) -> List[Dict[str, Any]]:
        """Split text into smaller chunks with metadata."""
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk_text = ""
        chunk_idx = start_index

        for sentence in sentences:
            if len(current_chunk_text) + len(sentence) <= self.chunk_size_chars:
                current_chunk_text += " " + sentence
            else:
                if current_chunk_text.strip():
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "text": self._normalize_text(current_chunk_text),
                        "metadata": {
                            "doc_title": doc_title,
                            "section_path": " > ".join(section_path),
                            "path": doc_path,
                            "chunk_index": chunk_idx,
                        },
                    })
                    chunk_idx += 1
                current_chunk_text = sentence

        if current_chunk_text.strip():
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": self._normalize_text(current_chunk_text),
                "metadata": {
                    "doc_title": doc_title,
                    "section_path": " > ".join(section_path),
                    "path": doc_path,
                    "chunk_index": chunk_idx,
                },
            })

        return chunks

    def _get_doc_metadata(self, file_path: str) -> Tuple[str, str]:
        doc_title = os.path.basename(file_path).replace("_", " ").replace(".pdf", "")
        doc_uri = f"file:///{os.path.abspath(file_path)}"
        return doc_title, doc_uri

    def _detect_headers_footers(self, doc: fitz.Document, num_pages_to_check: int = 5) -> List[fitz.Rect]:
        """Detect common header/footer text regions."""
        if len(doc) <= 1:
            return []

        page_indices = list(range(min(num_pages_to_check, len(doc))))
        if len(doc) > num_pages_to_check * 2:
            page_indices += list(range(len(doc) - num_pages_to_check, len(doc)))

        common_texts = defaultdict(int)
        text_positions = {}

        for i in page_indices:
            page = doc[i]
            header_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.15)
            footer_rect = fitz.Rect(0, page.rect.height * 0.85, page.rect.width, page.rect.height)

            for block in page.get_text("blocks"):
                block_rect = fitz.Rect(block[:4])
                block_text = self._normalize_text(block[4])
                if not block_text or len(block_text) > 100:
                    continue

                if block_rect.intersects(header_rect) or block_rect.intersects(footer_rect):
                    common_texts[block_text] += 1
                    if block_text not in text_positions:
                        text_positions[block_text] = block_rect

        repeated_texts = [text for text, count in common_texts.items() if count >= len(page_indices) * 0.7]
        return [text_positions[text] for text in repeated_texts]

    def _analyze_font_styles(self, doc: fitz.Document) -> Tuple[List[Dict], float]:
        """Analyze font sizes and styles to distinguish body text vs headings."""
        styles = defaultdict(int)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            style = (round(s["size"]), s["font"], s["flags"])
                            styles[style] += len(s["text"])

        if not styles:
            return [], 0.0

        sorted_styles = sorted(styles.items(), key=lambda item: item[1], reverse=True)
        body_style_size = sorted_styles[0][0][0]

        heading_styles = sorted([style for style, count in styles.items() if style[0] > body_style_size],
                                key=lambda s: s[0], reverse=True)

        return heading_styles, body_style_size

    def _handle_scanned_page(self, page: fitz.Page) -> str:
        """OCR for scanned PDFs."""
        if not OCR_AVAILABLE:
            return ""
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(image, lang=self.ocr_language)

    def _parse_with_styles(self, doc: fitz.Document, file_path: str) -> List[Dict]:
        """Fallback parser using font sizes and heuristics."""
        doc_title, doc_path = self._get_doc_metadata(file_path)
        heading_styles, body_style_size = self._analyze_font_styles(doc)
        header_footer_rects = self._detect_headers_footers(doc)

        chunks = []
        current_section_text = ""
        section_path = []
        chunk_index = 0

        for page in doc:
            if not page.get_text().strip():
                page_text = self._handle_scanned_page(page)
                blocks = [{"lines": [{"spans": [{"text": page_text, "size": body_style_size}]}]}]
            else:
                blocks = page.get_text("dict")["blocks"]

            for b in blocks:
                block_rect = fitz.Rect(b["bbox"])
                if any(block_rect.intersects(r) for r in header_footer_rects):
                    continue

                if "lines" not in b:
                    continue

                for l in b["lines"]:
                    span_sizes = [round(s["size"]) for s in l["spans"] if s["text"].strip()]
                    if not span_sizes:
                        continue
                    line_size = max(span_sizes)
                    line_text = "".join([s["text"] for s in l["spans"]])

                    is_heading = line_size > body_style_size
                    if is_heading:
                        if current_section_text.strip():
                            new_chunks = self._split_text_into_chunks(
                                current_section_text, doc_title, doc_path, section_path, chunk_index
                            )
                            chunks.extend(new_chunks)
                            chunk_index += len(new_chunks)
                        current_section_text = ""
                        heading_level = next((i for i, s in enumerate(heading_styles) if s[0] == line_size),
                                             len(heading_styles))
                        section_path = section_path[:heading_level]
                        section_path.append(self._normalize_text(line_text))
                    else:
                        current_section_text += " " + line_text

        if current_section_text.strip():
            new_chunks = self._split_text_into_chunks(
                current_section_text, doc_title, doc_path, section_path, chunk_index
            )
            chunks.extend(new_chunks)

        return chunks

    def parse(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        doc = fitz.open(file_path)
        chunks = self._parse_with_styles(doc, file_path)
        doc.close()
        return chunks


def structured_pdf_parser(file_path: str) -> List[Dict]:
    parser = UniversalPDFParser(chunk_size_chars=4000)
    return parser.parse(file_path)

#
# if __name__ == "__main__":
#     # Change the test file path as needed
#     test_pdf_path = "C:/Users/swapn/Downloads/Apache Tomcat 7.pdf"
#     # test_pdf_path = "C:/Users/swapn/Downloads/server_issues.pdf"
#     # test_pdf_path = "C:/Users/swapn/Downloads/EMTMC.pdf"
#
#     try:
#         parsed = structured_pdf_parser(test_pdf_path)
#
#         import json
#         output = {
#             "document": test_pdf_path,
#             "total_chunks": len(parsed),
#             "chunks": parsed
#         }
#
#         print(json.dumps(output, indent=2, ensure_ascii=False))
#
#         with open("parsed_output.json", "w", encoding="utf-8") as f:
#             json.dump(output, f, indent=2, ensure_ascii=False)
#
#         print(f"\n✅ Successfully parsed. Output saved to parsed_output.json")
#
#     except Exception as e:
#         print(f"❌ Error while parsing {test_pdf_path}: {e}")
