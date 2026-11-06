
import pymupdf

from app.rag.document_loader.pdf_loader import extract_text_from_pdf


def test_extract_text_from_pdf(tmp_path):
    # Create a temporary 3-page PDF
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        for page_number in range(1, 4):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"This is test content on page {page_number}."
            )

        document.save(pdf_path)

    # Extract PDF content
    result = extract_text_from_pdf(pdf_path)

    # Verify extraction
    assert len(result) == 3
    assert result[0]["page_number"] == 1
    assert result[1]["page_number"] == 2
    assert result[2]["page_number"] == 3

    assert "page 1" in result[0]["text"]
    assert "page 2" in result[1]["text"]
    assert "page 3" in result[2]["text"]


def test_file_not_found():
    try:
        extract_text_from_pdf("nonexistent.pdf")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
