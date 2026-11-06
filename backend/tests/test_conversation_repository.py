import uuid
import pytest

from app.database.session import SessionLocal
from app.repositories.conversation_repository import ConversationRepository


def test_conversation_repository_crud_lifecycle():
    db = SessionLocal()
    repo = ConversationRepository(db)
    conv = None

    try:
        # 1. Tạo cuộc hội thoại
        conv = repo.create_conversation(title="Test Phiên Chat Mới")
        assert conv.id is not None
        assert conv.title == "Test Phiên Chat Mới"

        # 2. Lấy chi tiết
        fetched = repo.get_conversation(conv.id)
        assert fetched is not None
        assert fetched.id == conv.id

        # 3. Thêm tin nhắn
        msg_user = repo.add_message(
            conversation_id=conv.id,
            role="user",
            content="Xin chào AI",
        )
        assert msg_user.id is not None
        assert msg_user.role == "user"

        msg_ai = repo.add_message(
            conversation_id=conv.id,
            role="assistant",
            content="Chào bạn, tôi có thể giúp gì? [Trang 1]",
            citations=[{"document_id": str(uuid.uuid4()), "page_number": 1, "snippet": "Snippet"}],
            is_grounded=True,
        )
        assert msg_ai.id is not None
        assert msg_ai.role == "assistant"

        # 4. Lấy danh sách tin nhắn theo thứ tự thời gian
        messages = repo.get_messages(conv.id)
        assert len(messages) == 2
        assert messages[0].id == msg_user.id
        assert messages[1].id == msg_ai.id

        # 5. Cập nhật tiêu đề cuộc hội thoại
        updated_conv = repo.update_title(conv, "Tiêu đề đã được cập nhật")
        assert updated_conv.title == "Tiêu đề đã được cập nhật"

        # 6. Lấy danh sách cuộc hội thoại
        conv_list = repo.list_conversations(skip=0, limit=10)
        assert len(conv_list) >= 1
        all_ids = [c.id for c in conv_list]
        assert conv.id in all_ids

    finally:
        if conv:
            # 7. Xóa cuộc hội thoại
            repo.delete_conversation(conv)
            assert repo.get_conversation(conv.id) is None
            assert len(repo.get_messages(conv.id)) == 0
        db.close()

