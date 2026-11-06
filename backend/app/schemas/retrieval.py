import uuid
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    id: str
    text: str
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    score: float = Field(..., ge=0.0, le=1.0, description="Độ tương đồng ngữ nghĩa (0.0 - 1.0)")
    distance: float | None = None


class Citation(BaseModel):
    document_id: uuid.UUID
    page_number: int
    chunk_index: int
    score: float
    snippet: str


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi hoặc cụm từ tìm kiếm")
    top_k: int = Field(5, ge=1, le=20, description="Số lượng đoạn văn bản cần lấy")
    document_id: uuid.UUID | None = Field(None, description="Lọc theo 1 tài liệu cụ thể")
    document_ids: list[uuid.UUID] | None = Field(None, description="Lọc theo danh sách tài liệu")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Ngưỡng tương đồng tối thiểu")


class RetrievalResponse(BaseModel):
    query: str
    total_retrieved: int
    chunks: list[RetrievedChunk]
    citations: list[Citation]
    context_text: str

