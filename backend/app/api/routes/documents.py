import uuid
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.document_service import delete_document_with_file
from app.services.upload_service import handle_pdf_upload

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload file PDF, lưu trữ an toàn, trích xuất & chia chunk,
    và lưu metadata vào PostgreSQL.
    """
    document = await handle_pdf_upload(file=file, db=db)
    return DocumentUploadResponse(
        message="Document uploaded and processed successfully.",
        document=DocumentResponse.model_validate(document),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
)
def get_documents(
    skip: int = Query(0, ge=0, description="Số lượng bản ghi bỏ qua"),
    limit: int = Query(20, ge=1, le=100, description="Số lượng bản ghi tối đa"),
    search: str | None = Query(None, description="Tìm kiếm theo tên file gốc"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    Task 2.7.1 & 2.7.6: Lấy danh sách tài liệu có phân trang và tìm kiếm theo tên.
    """
    repo = DocumentRepository(db)
    documents = repo.list_documents(skip=skip, limit=limit, search=search)
    total = repo.count_documents(search=search)
    return DocumentListResponse(
        total=total,
        items=[DocumentResponse.model_validate(doc) for doc in documents],
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
)
def get_document_detail(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    Task 2.7.2: Xem chi tiết một tài liệu theo ID.
    """
    repo = DocumentRepository(db)
    document = repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_document_status(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    """
    Task 2.7.3: Xem trạng thái xử lý của tài liệu theo ID.
    """
    repo = DocumentRepository(db)
    document = repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return DocumentStatusResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentDeleteResponse:
    """
    Task 2.7.4 & 2.7.5: Xóa tài liệu khỏi database và xóa file vật lý trên đĩa.
    """
    repo = DocumentRepository(db)
    document = repo.get_by_id(document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    delete_document_with_file(document=document, repo=repo)
    return DocumentDeleteResponse(
        message="Document deleted successfully.",
        id=document_id,
    )

