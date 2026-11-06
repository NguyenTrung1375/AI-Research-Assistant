import uuid
import pytest
from sqlalchemy.exc import IntegrityError

from app.database.session import SessionLocal
from app.models.conversation import Conversation, Message


def test_conversation_and_message_crud():
    db = SessionLocal()
    conversation = None

    try:
        # 1. Tạo conversation
        conversation = Conversation(
            title="Nghiên cứu về Attention Mechanism",
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        assert conversation.id is not None
        assert conversation.title == "Nghiên cứu về Attention Mechanism"

        # 2. Tạo message từ user
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content="Self-attention là gì?",
        )
        db.add(user_msg)

        # 3. Tạo message từ assistant kèm citations
        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Self-attention là cơ chế tính trọng số [Trang 4].",
            citations=[{"document_id": str(uuid.uuid4()), "page_number": 4, "snippet": "Self-attention..."}],
            is_grounded=True,
        )
        db.add(assistant_msg)
        db.commit()

        # 4. Kiểm tra messages liên kết
        db.refresh(conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"
        assert conversation.messages[1].citations[0]["page_number"] == 4

    finally:
        if conversation:
            # Test cascade delete: Xóa conversation phải tự động xóa messages
            conv_id = conversation.id
            db.delete(conversation)
            db.commit()

            # Xác nhận messages đã bị xóa sạch (Cascade delete)
            remaining_msgs = db.query(Message).filter(Message.conversation_id == conv_id).all()
            assert len(remaining_msgs) == 0

        db.close()


def test_message_role_check_constraint_rejects_invalid_role():
    db = SessionLocal()
    conversation = Conversation(title="Test Constraint")
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    try:
        invalid_msg = Message(
            conversation_id=conversation.id,
            role="hacker",  # Sai check constraint (chỉ cho phép 'user' hoặc 'assistant')
            content="Hello",
        )
        db.add(invalid_msg)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
    finally:
        db.delete(conversation)
        db.commit()
        db.close()

