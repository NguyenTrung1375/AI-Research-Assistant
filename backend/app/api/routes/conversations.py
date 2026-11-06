import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationDeleteResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    payload: ConversationCreateRequest,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """
    Task 6.3: Khởi tạo một phiên hội thoại mới (tùy chọn gắn với 1 tài liệu cụ thể).
    """
    repo = ConversationRepository(db)

    # Nếu có document_id, kiểm tra tài liệu có tồn tại không
    if payload.document_id:
        doc_repo = DocumentRepository(db)
        if doc_repo.get_by_id(payload.document_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {payload.document_id} not found.",
            )

    title = payload.title.strip() if payload.title and payload.title.strip() else "Cuộc hội thoại mới"
    conversation = repo.create_conversation(title=title, document_id=payload.document_id)
    return ConversationResponse.model_validate(conversation)


@router.get(
    "",
    response_model=ConversationListResponse,
    status_code=status.HTTP_200_OK,
)
def list_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ConversationListResponse:
    """
    Task 6.5: Lấy danh sách các cuộc hội thoại gần nhất.
    """
    repo = ConversationRepository(db)
    conversations = repo.list_conversations(skip=skip, limit=limit)
    return ConversationListResponse(
        total=len(conversations),
        items=[ConversationResponse.model_validate(c) for c in conversations],
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_200_OK,
)
def get_conversation_detail(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ConversationDetailResponse:
    """
    Task 6.5: Lấy toàn bộ lịch sử tin nhắn của một cuộc hội thoại.
    """
    repo = ConversationRepository(db)
    conversation = repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    messages = repo.get_messages(conversation_id)
    detail_data = ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        document_id=conversation.document_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[MessageResponse.model_validate(m) for m in messages],
    )
    return detail_data


@router.delete(
    "/{conversation_id}",
    response_model=ConversationDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ConversationDeleteResponse:
    """
    Task 6.6: Xóa một cuộc hội thoại (tự động xóa toàn bộ tin nhắn liên quan).
    """
    repo = ConversationRepository(db)
    conversation = repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    repo.delete_conversation(conversation)
    return ConversationDeleteResponse(
        message="Conversation deleted successfully.",
        id=conversation_id,
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_200_OK,
)
def send_chat_message(
    conversation_id: uuid.UUID,
    payload: SendMessageRequest,
    db: Session = Depends(get_db),
) -> SendMessageResponse:
    """
    Task 6.4 & 6.7: Gửi câu hỏi vào phiên chat, kích hoạt RAG Pipeline
    và lưu lại tin nhắn của cả User và Assistant.
    """
    chat_service = ChatService(db=db)
    try:
        user_msg, assistant_msg = chat_service.send_message(
            conversation_id=conversation_id,
            query=payload.content,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return SendMessageResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
    )

