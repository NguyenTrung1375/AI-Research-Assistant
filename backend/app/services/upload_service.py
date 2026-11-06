import uuid
from pathlib import Path

import pymupdf
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.rag.vector_store.chroma_store import get_vector_store
from app.repositories.document_repository import DocumentRepository
from app.services.document_processor import process_pdf

PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_original_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )
    clean_name = Path(filename.strip()).name
    if not clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )
    return clean_name[:255]


def validate_pdf_upload(
    filename: str,
    content_type: str | None,
    file_bytes: bytes,
    max_size_mb: int = settings.MAX_UPLOAD_SIZE_MB,
) -> None:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are allowed.",
        )

    allowed_mime_types = {"application/pdf", "application/x-pdf", "binary/octet-stream"}
    if content_type and content_type.lower() not in allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{content_type}'. Expected 'application/pdf'.",
        )

    file_size = len(file_bytes)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty.",
        )

    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of {max_size_mb} MB.",
        )

    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file format (missing PDF header).",
        )


def _get_pdf_page_count(file_path: Path) -> int:
    with pymupdf.open(file_path) as pdf_doc:
        return len(pdf_doc)


async def handle_pdf_upload(
    file: UploadFile,
    db: Session,
    upload_dir: str | Path | None = None,
) -> Document:
    original_filename = sanitize_original_filename(file.filename)
    file_bytes = await file.read()

    validate_pdf_upload(
        filename=original_filename,
        content_type=file.content_type,
        file_bytes=file_bytes,
        max_size_mb=settings.MAX_UPLOAD_SIZE_MB,
    )

    target_dir = Path(upload_dir) if upload_dir else Path(settings.UPLOAD_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid.uuid4().hex}.pdf"
    saved_file_path = target_dir / stored_filename

    saved_file_path.write_bytes(file_bytes)

    repo = DocumentRepository(db)
    document = repo.create(
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=str(saved_file_path),
        file_size=len(file_bytes),
        status="processing",
    )

    vector_store = get_vector_store()

    try:
        chunks = process_pdf(saved_file_path)
        if not chunks:
            raise ValueError("No extractable text found in the PDF document.")

        # Task 3.7: Tạo vector embedding và lưu trữ vào ChromaDB
        vector_store.add_chunks(document_id=document.id, chunks=chunks)

        page_count = _get_pdf_page_count(saved_file_path)
        document = repo.update_processing_result(
            document=document,
            page_count=page_count,
            chunk_count=len(chunks),
        )
        return document

    except ValueError as exc:
        vector_store.delete_by_document_id(document.id)
        repo.update_failed_status(document=document, error_message=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        vector_store.delete_by_document_id(document.id)
        repo.update_failed_status(
            document=document,
            error_message="Unexpected error during PDF processing.",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process PDF document.",
        ) from exc