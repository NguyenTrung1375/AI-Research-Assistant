import uuid

from app.database.session import SessionLocal
from app.repositories.document_repository import DocumentRepository


def test_document_repository_crud_lifecycle():
    db = SessionLocal()
    repo = DocumentRepository(db)
    created_doc = None

    try:
        stored_name = f"{uuid.uuid4().hex}.pdf"

        # 1. Test create
        created_doc = repo.create(
            original_filename="research_paper.pdf",
            stored_filename=stored_name,
            file_path=f"data/uploads/{stored_name}",
            file_size=102400,
        )
        assert created_doc.id is not None
        assert created_doc.original_filename == "research_paper.pdf"
        assert created_doc.status == "processing"
        assert created_doc.page_count is None

        # 2. Test get_by_id
        fetched_doc = repo.get_by_id(created_doc.id)
        assert fetched_doc is not None
        assert fetched_doc.stored_filename == stored_name

        # 3. Test update_processing_result
        completed_doc = repo.update_processing_result(
            document=fetched_doc,
            page_count=12,
            chunk_count=45,
        )
        assert completed_doc.status == "completed"
        assert completed_doc.page_count == 12
        assert completed_doc.chunk_count == 45
        assert completed_doc.error_message is None

        # 4. Test update_failed_status
        failed_doc = repo.update_failed_status(
            document=completed_doc,
            error_message="Corrupted PDF stream",
        )
        assert failed_doc.status == "failed"
        assert failed_doc.error_message == "Corrupted PDF stream"

        # 5. Test delete
        doc_id = failed_doc.id
        repo.delete(failed_doc)
        created_doc = None
        assert repo.get_by_id(doc_id) is None

    finally:
        if created_doc is not None:
            repo.delete(created_doc)
        db.close()