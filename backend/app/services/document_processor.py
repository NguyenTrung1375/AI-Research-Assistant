
from pathlib import Path

from app.rag.document_loader.pdf_loader import extract_text_from_pdf
from app.rag.document_loader.text_cleaner import clean_text
from app.rag.text_splitter.chunker import chunk_pages


def process_pdf(
    file_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Process a PDF through extraction, cleaning and chunking.

    Returns:
        List of chunks containing:
        - page_number
        - chunk_index
        - text
    """

    # Step 1: Extract text page by page
    pages = extract_text_from_pdf(file_path)

    # Step 2: Clean text while preserving page metadata
    cleaned_pages = []

    for page in pages:
        cleaned_text = clean_text(page["text"])

        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": cleaned_text,
        })

    # Step 3: Split pages into overlapping chunks
    chunks = chunk_pages(
        pages=cleaned_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunks
