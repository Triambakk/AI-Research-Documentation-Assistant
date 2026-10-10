import pymupdf


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract non-empty text from a PDF while preserving page numbers."""
    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

    return pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """Split page text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text = page["text"].strip()

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                })
                chunk_id += 1

    return chunks
