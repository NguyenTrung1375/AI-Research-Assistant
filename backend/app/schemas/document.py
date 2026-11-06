import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    stored_filename: str
    file_size: int
    page_count: int | None = None
    chunk_count: int | None = None
    status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentResponse


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentResponse]
    skip: int = 0
    limit: int = 20


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    updated_at: datetime


class DocumentDeleteResponse(BaseModel):
    message: str
    id: uuid.UUID

