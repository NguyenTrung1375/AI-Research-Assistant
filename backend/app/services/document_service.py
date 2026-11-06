import logging
from pathlib import Path

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


def delete_physical_file(file_path: str | Path) -> bool:
    """
    Xóa file vật lý trên ổ đĩa an toàn (Task 2.7.5).
    Trả về True nếu xóa thành công, False nếu file không tồn tại hoặc lỗi.
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
    except OSError as exc:
        logger.warning("Failed to delete physical file %s: %s", file_path, exc)
        return False


def delete_document_with_file(document: Document, repo: DocumentRepository) -> None:
    """
    Xóa file vật lý (Task 2.7.5), xóa vectors trong ChromaDB (Task 3.8),
    và xóa record tài liệu trong PostgreSQL (Task 2.7.4).
    """
    if document.file_path:
        delete_physical_file(document.file_path)

    # Task 3.8: Xóa vectors khỏi ChromaDB, tránh vector mồ côi
    try:
        from app.rag.vector_store.chroma_store import get_vector_store

        vector_store = get_vector_store()
        vector_store.delete_by_document_id(document.id)
    except Exception as exc:
        logger.warning("Failed to delete vectors from ChromaDB for doc %s: %s", document.id, exc)

    repo.delete(document)

