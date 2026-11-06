import uuid
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.session import SessionLocal
from app.main import app
from app.models.document import Document

client = TestClient(app)


def _create_valid_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _cleanup_document_by_id(doc_id: str | uuid.UUID) -> None:
    db = SessionLocal()
    try:
        parsed_id = uuid.UUID(str(doc_id))
        doc = db.get(Document, parsed_id)
        if doc:
            file_path = Path(doc.file_path)
            if file_path.exists():
                file_path.unlink()
            db.delete(doc)
            db.commit()
    finally:
        db.close()


def _cleanup_document_by_original_name(original_filename: str) -> None:
    db = SessionLocal()
    try:
        docs = (
            db.query(Document)
            .filter(Document.original_filename == original_filename)
            .all()
        )
        for doc in docs:
            file_path = Path(doc.file_path)
            if file_path.exists():
                file_path.unlink()
            db.delete(doc)
        db.commit()
    finally:
        db.close()


def test_upload_valid_pdf_success():
    pdf_bytes = _create_valid_pdf_bytes(
        [
            "Page 1: Introduction to Retrieval-Augmented Generation systems.",
            "Page 2: Vector databases and semantic search architecture.",
        ]
    )
    created_id = None

    try:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("../unsafe_path/sample_paper.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()

        assert data["message"] == "Document uploaded and processed successfully."
        doc_data = data["document"]
        created_id = doc_data["id"]

        # Kiểm tra tên file gốc đã được làm sạch và tên lưu trữ là UUID
        assert doc_data["original_filename"] == "sample_paper.pdf"
        assert doc_data["stored_filename"] != "sample_paper.pdf"
        assert doc_data["stored_filename"].endswith(".pdf")
        assert doc_data["file_size"] == len(pdf_bytes)
        assert doc_data["page_count"] == 2
        assert doc_data["chunk_count"] >= 2
        assert doc_data["status"] == "completed"
        assert doc_data["error_message"] is None

        # Kiểm tra bản ghi trong PostgreSQL và file vật lý
        db = SessionLocal()
        try:
            db_doc = db.get(Document, uuid.UUID(created_id))
            assert db_doc is not None
            assert Path(db_doc.file_path).exists()
        finally:
            db.close()

    finally:
        if created_id:
            _cleanup_document_by_id(created_id)


def test_upload_non_pdf_file_rejected():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"Plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]


def test_upload_fake_pdf_header_rejected():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"NOT_A_REAL_PDF_CONTENT", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid PDF file format" in response.json()["detail"]


def test_upload_oversized_pdf_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)
    oversized_bytes = b"%PDF-1.4\n" + (b"0" * (1024 * 1024 + 100))

    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.pdf", oversized_bytes, "application/pdf")},
    )
    assert response.status_code == 413
    assert "exceeds the maximum limit" in response.json()["detail"]


def test_upload_corrupted_pdf_marks_status_failed():
    unique_name = f"corrupted_{uuid.uuid4().hex[:8]}.pdf"
    corrupted_bytes = b"%PDF-1.7\nCorrupted stream content that PyMuPDF cannot parse"

    try:
        response = client.post(
            "/api/documents/upload",
            files={"file": (unique_name, corrupted_bytes, "application/pdf")},
        )
        assert response.status_code == 422

        db = SessionLocal()
        try:
            db_doc = (
                db.query(Document)
                .filter(Document.original_filename == unique_name)
                .first()
            )
            assert db_doc is not None
            assert db_doc.status == "failed"
            assert db_doc.error_message is not None
        finally:
            db.close()
    finally:
        _cleanup_document_by_original_name(unique_name)


def test_upload_empty_text_pdf_marks_status_failed():
    unique_name = f"blank_{uuid.uuid4().hex[:8]}.pdf"
    blank_pdf_bytes = _create_valid_pdf_bytes(["", ""])

    try:
        response = client.post(
            "/api/documents/upload",
            files={"file": (unique_name, blank_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 422
        assert "No extractable text" in response.json()["detail"]

        db = SessionLocal()
        try:
            db_doc = (
                db.query(Document)
                .filter(Document.original_filename == unique_name)
                .first()
            )
            assert db_doc is not None
            assert db_doc.status == "failed"
            assert "No extractable text" in db_doc.error_message
        finally:
            db.close()
    finally:
        _cleanup_document_by_original_name(unique_name)