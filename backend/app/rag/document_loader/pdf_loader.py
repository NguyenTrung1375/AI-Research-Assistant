
from pathlib import Path

import pymupdf


def extract_text_from_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF file page by page.

    Returns:
        A list of dictionaries containing page_number and text.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    try:
        with pymupdf.open(pdf_path) as document:

            if document.is_encrypted:
                raise ValueError("Encrypted PDF files are not supported.")

            if len(document) == 0:
                raise ValueError("The PDF document contains no pages.")

            extracted_pages = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()

                extracted_pages.append({
                    "page_number": page_number,
                    "text": page_text,
                })

            return extracted_pages

    except (pymupdf.FileDataError, RuntimeError) as error:
        raise ValueError(f"Unable to read PDF file: {error}") from error
