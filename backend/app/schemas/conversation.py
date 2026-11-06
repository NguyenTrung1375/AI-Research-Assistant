import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]] | None = None
    is_grounded: bool = True
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    document_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime


class ConversationDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    document_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(None, max_length=255, description="Tiêu đề phiên chat (tùy chọn)")
    document_id: uuid.UUID | None = Field(None, description="Gắn phiên chat với 1 tài liệu cụ thể")


class ConversationListResponse(BaseModel):
    total: int
    items: list[ConversationResponse]


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Nội dung câu hỏi của người dùng")
    top_k: int = Field(5, ge=1, le=20, description="Số lượng đoạn văn bản trích xuất tối đa")
    min_score: float = Field(0.3, ge=0.0, le=1.0, description="Ngưỡng tương đồng ngữ nghĩa tối thiểu")


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class ConversationDeleteResponse(BaseModel):
    message: str
    id: uuid.UUID

