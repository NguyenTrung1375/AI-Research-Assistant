import io
import uuid
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.session import SessionLocal
from app.main import app
from app.models.document import Document
from app.rag.vector_store.chroma_store import get_vector_store

client = TestClient(app)


def _create_sample_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Artificial intelligence and deep learning in modern research.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Retrieval augmented generation architectures for document analysis.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_full_document_lifecycle_integration():
    """
    Task 2.7.7: Integration Test bao quát toàn bộ vòng đời tài liệu (Upload -> List -> Search -> Detail -> Status -> Delete).
    """
    suffix = uuid.uuid4().hex[:6]
    original_name = f"lifecycle_research_{suffix}.pdf"
    pdf_bytes = _create_sample_pdf_bytes()

    doc_id = None
    saved_file_path = None

    try:
        # 1. Upload PDF
        upload_res = client.post(
            "/api/documents/upload",
            files={"file": (original_name, io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert upload_res.status_code == 201
        upload_data = upload_res.json()
        assert upload_data["message"] == "Document uploaded and processed successfully."

        doc_info = upload_data["document"]
        doc_id = doc_info["id"]
        saved_file_path = Path(settings.UPLOAD_DIR) / doc_info["stored_filename"]

        assert doc_info["original_filename"] == original_name
        assert doc_info["status"] == "completed"
        assert doc_info["page_count"] == 2
        assert doc_info["chunk_count"] >= 1
        assert saved_file_path.exists()

        # Task 3.7: Xác nhận vectors đã được lưu vào ChromaDB
        vector_store = get_vector_store()
        assert vector_store.count_by_document_id(doc_id) == doc_info["chunk_count"]

        # 2. List documents
        list_res = client.get("/api/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] >= 1
        all_ids = [item["id"] for item in list_data["items"]]
        assert doc_id in all_ids

        # 3. Search documents
        search_res = client.get(f"/api/documents?search=lifecycle_research_{suffix}")
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert search_data["total"] == 1
        assert search_data["items"][0]["id"] == doc_id

        # 4. Get detail
        detail_res = client.get(f"/api/documents/{doc_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["id"] == doc_id
        assert detail_data["page_count"] == 2
        assert detail_data["chunk_count"] == doc_info["chunk_count"]

        # 5. Get status
        status_res = client.get(f"/api/documents/{doc_id}/status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["id"] == doc_id
        assert status_data["status"] == "completed"
        assert status_data["page_count"] == 2

        # 6. Delete document
        delete_res = client.delete(f"/api/documents/{doc_id}")
        assert delete_res.status_code == 200
        assert delete_res.json()["message"] == "Document deleted successfully."

        # 7. Xác nhận file vật lý đã bị xóa
        assert not saved_file_path.exists()

        # Task 3.8: Xác nhận toàn bộ vectors trong ChromaDB đã bị xóa
        assert vector_store.count_by_document_id(doc_id) == 0

        # 8. Xác nhận record trong DB đã bị xóa
        assert client.get(f"/api/documents/{doc_id}").status_code == 404
        assert client.get(f"/api/documents/{doc_id}/status").status_code == 404

        # 9. Xác nhận không còn xuất hiện trong search
        verify_search = client.get(f"/api/documents?search=lifecycle_research_{suffix}")
        assert verify_search.status_code == 200
        assert verify_search.json()["total"] == 0

    finally:
        # Cleanup dự phòng nếu test fail giữa chừng
        if doc_id:
            db = SessionLocal()
            try:
                doc = db.get(Document, uuid.UUID(doc_id))
                if doc:
                    db.delete(doc)
                    db.commit()
            finally:
                db.close()
        if saved_file_path and saved_file_path.exists():
            saved_file_path.unlink()


def test_delete_handles_already_missing_physical_file_gracefully(tmp_path):
    """
    Task 2.7.7: Đảm bảo xóa tài liệu không bị crash 500 nếu file vật lý bị mất trước đó.
    """
    db = SessionLocal()
    missing_file = tmp_path / "ghost.pdf"
    doc = Document(
        original_filename="ghost.pdf",
        stored_filename="ghost.pdf",
        file_path=str(missing_file),
        file_size=1234,
        status="completed",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()

    try:
        assert not missing_file.exists()
        res = client.delete(f"/api/documents/{doc_id}")
        assert res.status_code == 200
        assert res.json()["message"] == "Document deleted successfully."

        # DB đã xóa thành công
        assert client.get(f"/api/documents/{doc_id}").status_code == 404
    finally:
        db_cleanup = SessionLocal()
        try:
            d = db_cleanup.get(Document, doc_id)
            if d:
                db_cleanup.delete(d)
                db_cleanup.commit()
        finally:
            db_cleanup.close()
