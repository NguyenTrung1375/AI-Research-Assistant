import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository quản lý các thao tác truy vấn cho Conversation và Message."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_conversation(
        self,
        title: str = "Cuộc hội thoại mới",
        document_id: uuid.UUID | None = None,
    ) -> Conversation:
        conversation = Conversation(
            title=title,
            document_id=document_id,
        )
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        return self.db.execute(stmt).scalar_one_or_none()

    def list_conversations(self, skip: int = 0, limit: int = 50) -> list[Conversation]:
        stmt = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def update_title(self, conversation: Conversation, title: str) -> Conversation:
        conversation.title = title[:255]
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def delete_conversation(self, conversation: Conversation) -> None:
        self.db.delete(conversation)
        self.db.commit()

    def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        is_grounded: bool = True,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            is_grounded=is_grounded,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_messages(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

