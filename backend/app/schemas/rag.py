import uuid
from pydantic import BaseModel, Field

from app.schemas.retrieval import Citation, RetrievedChunk


class RAGQuestionRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi nghiên cứu của người dùng")
    document_id: uuid.UUID | None = Field(None, description="Lọc theo 1 tài liệu cụ thể")
    document_ids: list[uuid.UUID] | None = Field(None, description="Lọc theo danh sách tài liệu")
    top_k: int = Field(5, ge=1, le=20, description="Số lượng đoạn ngữ cảnh trích xuất")
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Ngưỡng tương đồng tối thiểu (mặc định 0.3)")


class RAGAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    context_chunks: list[RetrievedChunk]
    is_grounded: bool

