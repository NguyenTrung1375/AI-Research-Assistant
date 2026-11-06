
import pymupdf

from app.services.document_processor import process_pdf


def test_process_pdf_pipeline(tmp_path):
    pdf_path = tmp_path / "research.pdf"

    # Create a sample PDF with 2 pages
    with pymupdf.open() as document:
        for page_number in range(1, 3):
            page = document.new_page()

            content = " ".join(
                f"word{i}" for i in range(1, 601)
            )

            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 780),
                content,
                fontsize=8,
            )

        document.save(pdf_path)

    # Process PDF
    chunks = process_pdf(
        pdf_path,
        chunk_size=500,
        chunk_overlap=80,
    )

    # Each page should produce 2 chunks
    assert len(chunks) == 4

    # Verify page metadata
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 1
    assert chunks[2]["page_number"] == 2
    assert chunks[3]["page_number"] == 2

    # Verify chunk indexes
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[2]["chunk_index"] == 0
    assert chunks[3]["chunk_index"] == 1

    # Verify extracted content
    assert "word1" in chunks[0]["text"]
    assert "word600" in chunks[1]["text"]
