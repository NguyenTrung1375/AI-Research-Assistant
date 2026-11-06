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
    Xóa file vật lý và xóa record tài liệu trong database.
    """
    if document.file_path:
        delete_physical_file(document.file_path)
    repo.delete(document)

