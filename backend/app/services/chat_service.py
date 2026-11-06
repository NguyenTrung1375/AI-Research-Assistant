import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message
from app.rag.generation.rag_orchestrator import RAGOrchestrator, get_rag_orchestrator
from app.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


class ChatService:
    """
    Deep Module điều phối quy trình hội thoại (Conversation Management):
      - Nhận câu hỏi và lưu tin nhắn User
      - Tự động sinh tiêu đề cho phiên chat mới
      - Giới hạn phạm vi tài liệu (nếu phiên chat liên kết với 1 document_id)
      - Gọi RAGOrchestrator để lấy câu trả lời và trích dẫn số trang
      - Lưu tin nhắn Assistant kèm citations và trạng thái is_grounded
    """

    def __init__(
        self,
        db: Session,
        orchestrator: RAGOrchestrator | None = None,
        repo: ConversationRepository | None = None,
    ) -> None:
        self.db = db
        self.repo = repo or ConversationRepository(db)
        self.orchestrator = orchestrator or get_rag_orchestrator()

    def send_message(
        self,
        conversation_id: uuid.UUID,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> tuple[Message, Message]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        conversation = self.repo.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # 1. Lưu tin nhắn của người dùng
        user_message = self.repo.add_message(
            conversation_id=conversation_id,
            role="user",
            content=cleaned_query,
        )

        # 2. Tự động cập nhật tiêu đề nếu là tin nhắn đầu tiên hoặc tiêu đề mặc định
        if conversation.title in ("Cuộc hội thoại mới", "", None):
            short_title = cleaned_query[:60] + ("..." if len(cleaned_query) > 60 else "")
            self.repo.update_title(conversation, short_title)

        # 3. Gọi RAG Orchestrator (giới hạn theo document_id nếu phiên chat có liên kết)
        rag_answer = self.orchestrator.answer_question(
            query=cleaned_query,
            document_id=conversation.document_id,
            top_k=top_k,
            min_score=min_score,
        )

        # 4. Định dạng citations lưu vào JSON
        citations_data: list[dict[str, Any]] = []
        for citation in rag_answer.citations:
            citations_data.append({
                "document_id": str(citation.document_id),
                "page_number": int(citation.page_number),
                "chunk_index": int(citation.chunk_index),
                "score": float(citation.score),
                "snippet": citation.snippet,
            })

        # 5. Lưu tin nhắn của Assistant
        assistant_message = self.repo.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_answer.answer,
            citations=citations_data,
            is_grounded=rag_answer.is_grounded,
        )

        return user_message, assistant_message

