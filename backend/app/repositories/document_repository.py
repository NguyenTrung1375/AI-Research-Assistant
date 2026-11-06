import uuid
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    """Repository quản lý các thao tác database cho bảng documents."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        original_filename: str,
        stored_filename: str,
        file_path: str,
        file_size: int,
        status: str = "processing",
    ) -> Document:
        document = Document(
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_size=file_size,
            status=status,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        search: str | None = None,
    ) -> list[Document]:
        stmt = select(Document)
        if search and search.strip():
            stmt = stmt.where(Document.original_filename.ilike(f"%{search.strip()}%"))
        stmt = stmt.order_by(Document.created_at.desc()).offset(skip).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count_documents(self, search: str | None = None) -> int:
        stmt = select(func.count()).select_from(Document)
        if search and search.strip():
            stmt = stmt.where(Document.original_filename.ilike(f"%{search.strip()}%"))
        return int(self.db.execute(stmt).scalar_one())

    def update_processing_result(
        self,
        document: Document,
        page_count: int,
        chunk_count: int,
    ) -> Document:
        document.page_count = page_count
        document.chunk_count = chunk_count
        document.status = "completed"
        document.error_message = None
        self.db.commit()
        self.db.refresh(document)
        return document

    def update_failed_status(
        self,
        document: Document,
        error_message: str,
    ) -> Document:
        document.status = "failed"
        document.error_message = error_message
        self.db.commit()
        self.db.refresh(document)
        return document

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.commit()