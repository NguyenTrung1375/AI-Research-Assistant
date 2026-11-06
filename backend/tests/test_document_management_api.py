import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.repositories.document_repository import DocumentRepository

client = TestClient(app)


def test_list_documents_returns_items_and_total():
    db = SessionLocal()
    repo = DocumentRepository(db)
    doc1 = None
    doc2 = None

    try:
        doc1 = repo.create(
            original_filename="paper_1.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/paper_1.pdf",
            file_size=1024,
            status="completed",
        )
        doc2 = repo.create(
            original_filename="paper_2.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/paper_2.pdf",
            file_size=2048,
            status="completed",
        )

        response = client.get("/api/documents")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] >= 2
        returned_ids = [item["id"] for item in data["items"]]
        assert str(doc1.id) in returned_ids
        assert str(doc2.id) in returned_ids

    finally:
        if doc1:
            repo.delete(doc1)
        if doc2:
            repo.delete(doc2)
        db.close()


def test_get_document_detail_success():
    db = SessionLocal()
    repo = DocumentRepository(db)
    doc = None

    try:
        doc = repo.create(
            original_filename="deep_learning.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/deep_learning.pdf",
            file_size=4096,
            status="processing",
        )
        doc = repo.update_processing_result(
            document=doc,
            page_count=10,
            chunk_count=35,
        )

        response = client.get(f"/api/documents/{doc.id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == str(doc.id)
        assert data["original_filename"] == "deep_learning.pdf"
        assert data["file_size"] == 4096
        assert data["page_count"] == 10
        assert data["chunk_count"] == 35
        assert data["status"] == "completed"

    finally:
        if doc:
            repo.delete(doc)
        db.close()


def test_get_document_detail_not_found():
    non_existent_id = uuid.uuid4()
    response = client.get(f"/api/documents/{non_existent_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_get_document_status_success():
    db = SessionLocal()
    repo = DocumentRepository(db)
    doc = None

    try:
        doc = repo.create(
            original_filename="status_test.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/status_test.pdf",
            file_size=2048,
            status="processing",
        )

        # 1. Trạng thái ban đầu: processing
        response = client.get(f"/api/documents/{doc.id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc.id)
        assert data["status"] == "processing"
        assert data["page_count"] is None
        assert data["chunk_count"] is None
        assert data["error_message"] is None
        assert "updated_at" in data

        # 2. Cập nhật sang completed
        repo.update_processing_result(doc, page_count=5, chunk_count=18)
        response_completed = client.get(f"/api/documents/{doc.id}/status")
        assert response_completed.status_code == 200
        data_completed = response_completed.json()
        assert data_completed["status"] == "completed"
        assert data_completed["page_count"] == 5
        assert data_completed["chunk_count"] == 18
        assert data_completed["error_message"] is None

    finally:
        if doc:
            repo.delete(doc)
        db.close()


def test_get_document_status_failed():
    db = SessionLocal()
    repo = DocumentRepository(db)
    doc = None

    try:
        doc = repo.create(
            original_filename="corrupted.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/corrupted.pdf",
            file_size=1024,
            status="processing",
        )

        repo.update_failed_status(doc, error_message="Failed to extract text from PDF.")
        response = client.get(f"/api/documents/{doc.id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc.id)
        assert data["status"] == "failed"
        assert data["error_message"] == "Failed to extract text from PDF."

    finally:
        if doc:
            repo.delete(doc)
        db.close()


def test_get_document_status_not_found():
    non_existent_id = uuid.uuid4()
    response = client.get(f"/api/documents/{non_existent_id}/status")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_delete_document_success_and_removes_physical_file(tmp_path):
    db = SessionLocal()
    repo = DocumentRepository(db)

    # Tạo file vật lý giả lập
    test_file = tmp_path / "test_delete_me.pdf"
    test_file.write_bytes(b"%PDF-1.4 dummy content")
    assert test_file.exists()

    doc = repo.create(
        original_filename="test_delete_me.pdf",
        stored_filename=test_file.name,
        file_path=str(test_file),
        file_size=len(test_file.read_bytes()),
        status="completed",
    )

    try:
        response = client.delete(f"/api/documents/{doc.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document deleted successfully."
        assert data["id"] == str(doc.id)

        # 1. File vật lý phải bị xóa khỏi ổ đĩa (Task 2.7.5)
        assert not test_file.exists()

        # 2. Record trong DB phải không còn tồn tại (Task 2.7.4)
        get_response = client.get(f"/api/documents/{doc.id}")
        assert get_response.status_code == 404

    finally:
        # Dọn dẹp nếu test fail giữa chừng
        cleanup_doc = repo.get_by_id(doc.id)
        if cleanup_doc:
            repo.delete(cleanup_doc)
        if test_file.exists():
            test_file.unlink()
        db.close()


def test_delete_document_not_found():
    non_existent_id = uuid.uuid4()
    response = client.delete(f"/api/documents/{non_existent_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_list_documents_pagination_and_search():
    db = SessionLocal()
    repo = DocumentRepository(db)
    docs = []

    try:
        suffix = uuid.uuid4().hex[:6]
        d1 = repo.create(
            original_filename=f"quantum_doc_{suffix}.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/quantum.pdf",
            file_size=1024,
            status="completed",
        )
        d2 = repo.create(
            original_filename=f"quantum_paper_{suffix}.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/neural.pdf",
            file_size=2048,
            status="completed",
        )
        d3 = repo.create(
            original_filename=f"economics_market_{suffix}.pdf",
            stored_filename=f"{uuid.uuid4().hex}.pdf",
            file_path="data/uploads/market.pdf",
            file_size=3072,
            status="completed",
        )
        docs.extend([d1, d2, d3])

        # 1. Test Search theo từ khóa "quantum"
        res_search = client.get(f"/api/documents?search=quantum_{suffix}")
        assert res_search.status_code == 200
        data_search = res_search.json()
        assert data_search["total"] == 0  # "quantum_doc_..." không chứa "quantum_<suffix>"

        # Test search với từ khóa khớp chính xác cả 2 tài liệu quantum: "quantum"
        res_quantum = client.get(f"/api/documents?search=quantum")
        assert res_quantum.status_code == 200
        data_quantum = res_quantum.json()
        assert data_quantum["total"] >= 2
        quantum_ids = [item["id"] for item in data_quantum["items"]]
        assert str(d1.id) in quantum_ids
        assert str(d2.id) in quantum_ids
        assert str(d3.id) not in quantum_ids

        # 2. Test Search không khớp
        res_empty = client.get(f"/api/documents?search=nonexistent_xyz_{suffix}")
        assert res_empty.status_code == 200
        assert res_empty.json()["total"] == 0
        assert res_empty.json()["items"] == []

        # 3. Test Pagination (limit=1, skip=0)
        res_page1 = client.get(f"/api/documents?search={suffix}&skip=0&limit=1")
        assert res_page1.status_code == 200
        data_page1 = res_page1.json()
        assert data_page1["total"] == 3
        assert len(data_page1["items"]) == 1
        assert data_page1["skip"] == 0
        assert data_page1["limit"] == 1

        # Test Pagination (limit=1, skip=1)
        res_page2 = client.get(f"/api/documents?search={suffix}&skip=1&limit=1")
        assert res_page2.status_code == 200
        data_page2 = res_page2.json()
        assert len(data_page2["items"]) == 1
        assert data_page2["items"][0]["id"] != data_page1["items"][0]["id"]

    finally:
        for d in docs:
            existing = repo.get_by_id(d.id)
            if existing:
                repo.delete(existing)
        db.close()

