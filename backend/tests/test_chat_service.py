import uuid
from unittest.mock import MagicMock
import pytest

from app.database.session import SessionLocal
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.rag import RAGAnswer
from app.schemas.retrieval import Citation
from app.services.chat_service import ChatService


def test_chat_service_send_message_stores_both_messages_and_updates_title():
    db = SessionLocal()
    repo = ConversationRepository(db)
    conv = repo.create_conversation(title="Cuộc hội thoại mới")

    doc_id = uuid.uuid4()
    mock_orchestrator = MagicMock()
    mock_orchestrator.answer_question.return_value = RAGAnswer(
        query="Self-attention là gì?",
        answer="Self-attention là cơ chế tính toán trọng số ngữ cảnh [Trang 3].",
        citations=[
            Citation(
                document_id=doc_id,
                page_number=3,
                chunk_index=0,
                score=0.91,
                snippet="Self-attention tính toán...",
            )
        ],
        context_chunks=[],
        is_grounded=True,
    )

    try:
        service = ChatService(db=db, orchestrator=mock_orchestrator, repo=repo)
        user_msg, assistant_msg = service.send_message(
            conversation_id=conv.id,
            query="Self-attention là gì?",
        )

        # 1. Kiểm tra tin nhắn của User
        assert user_msg.role == "user"
        assert user_msg.content == "Self-attention là gì?"

        # 2. Kiểm tra tin nhắn của Assistant
        assert assistant_msg.role == "assistant"
        assert "Self-attention là cơ chế" in assistant_msg.content
        assert len(assistant_msg.citations) == 1
        assert assistant_msg.citations[0]["page_number"] == 3
        assert assistant_msg.is_grounded is True

        # 3. Kiểm tra tự động cập nhật tiêu đề cuộc hội thoại theo câu hỏi đầu tiên
        refreshed_conv = repo.get_conversation(conv.id)
        assert refreshed_conv.title == "Self-attention là gì?"

        # 4. Kiểm tra lịch sử cuộc hội thoại có 2 tin nhắn
        messages = repo.get_messages(conv.id)
        assert len(messages) == 2

    finally:
        repo.delete_conversation(conv)
        db.close()


def test_chat_service_scopes_to_linked_document():
    db = SessionLocal()
    repo = ConversationRepository(db)
    doc_repo = DocumentRepository(db)

    # Tạo document hợp lệ để thỏa mãn foreign key constraint
    doc = doc_repo.create(
        original_filename="sample_paper.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        file_path="data/uploads/sample_paper.pdf",
        file_size=1024,
        status="completed",
    )
    doc_id = doc.id
    conv = repo.create_conversation(title="Phân tích bài báo X", document_id=doc_id)

    mock_orchestrator = MagicMock()
    mock_orchestrator.answer_question.return_value = RAGAnswer(
        query="Nội dung chính?",
        answer="Nội dung [Trang 1].",
        citations=[],
        context_chunks=[],
        is_grounded=True,
    )

    try:
        service = ChatService(db=db, orchestrator=mock_orchestrator, repo=repo)
        service.send_message(conversation_id=conv.id, query="Nội dung chính?")

        # Đảm bảo RAGOrchestrator được gọi với đúng document_id gắn với cuộc hội thoại
        mock_orchestrator.answer_question.assert_called_once()
        call_kwargs = mock_orchestrator.answer_question.call_args.kwargs
        assert call_kwargs["document_id"] == doc_id

    finally:
        repo.delete_conversation(conv)
        doc_repo.delete(doc)
        db.close()


def test_chat_service_raises_error_when_conversation_not_found():
    db = SessionLocal()
    service = ChatService(db=db, orchestrator=MagicMock())
    try:
        with pytest.raises(ValueError, match="Conversation not found"):
            service.send_message(conversation_id=uuid.uuid4(), query="Hello")
    finally:
        db.close()
