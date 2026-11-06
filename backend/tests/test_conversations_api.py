import uuid
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.conversation import Conversation
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.rag import RAGAnswer
from app.schemas.retrieval import Citation

client = TestClient(app)


def test_create_and_list_conversations():
    # 1. Tạo conversation không truyền title (dùng default)
    res_create = client.post("/api/conversations", json={})
    assert res_create.status_code == 201
    data_create = res_create.json()
    conv_id = data_create["id"]
    assert data_create["title"] == "Cuộc hội thoại mới"

    try:
        # 2. List conversations
        res_list = client.get("/api/conversations")
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert data_list["total"] >= 1
        all_ids = [item["id"] for item in data_list["items"]]
        assert conv_id in all_ids

        # 3. Get detail
        res_detail = client.get(f"/api/conversations/{conv_id}")
        assert res_detail.status_code == 200
        data_detail = res_detail.json()
        assert data_detail["id"] == conv_id
        assert data_detail["messages"] == []

    finally:
        # 4. Delete conversation
        res_del = client.delete(f"/api/conversations/{conv_id}")
        assert res_del.status_code == 200
        assert client.get(f"/api/conversations/{conv_id}").status_code == 404


def test_create_conversation_with_invalid_document_id():
    res = client.post("/api/conversations", json={"document_id": str(uuid.uuid4())})
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_send_message_in_conversation(monkeypatch):
    db = SessionLocal()
    repo = ConversationRepository(db)
    conv = repo.create_conversation(title="Cuộc hội thoại mới")
    conv_id = str(conv.id)
    db.close()

    doc_id = uuid.uuid4()
    mock_orchestrator = MagicMock()
    mock_orchestrator.answer_question.return_value = RAGAnswer(
        query="Self-attention là gì?",
        answer="Self-attention là cơ chế tính toán trọng số theo ngữ cảnh [Trang 5].",
        citations=[
            Citation(
                document_id=doc_id,
                page_number=5,
                chunk_index=0,
                score=0.93,
                snippet="Self-attention...",
            )
        ],
        context_chunks=[],
        is_grounded=True,
    )

    monkeypatch.setattr(
        "app.services.chat_service.get_rag_orchestrator",
        lambda: mock_orchestrator,
    )

    try:
        res = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Self-attention là gì?"},
        )
        assert res.status_code == 200
        data = res.json()

        # Kiểm tra user message
        assert data["user_message"]["role"] == "user"
        assert data["user_message"]["content"] == "Self-attention là gì?"

        # Kiểm tra assistant message
        assert data["assistant_message"]["role"] == "assistant"
        assert "Self-attention là cơ chế" in data["assistant_message"]["content"]
        assert len(data["assistant_message"]["citations"]) == 1
        assert data["assistant_message"]["citations"][0]["page_number"] == 5

        # Kiểm tra chi tiết conversation đã lưu 2 tin nhắn và tự đổi title
        res_detail = client.get(f"/api/conversations/{conv_id}")
        assert res_detail.status_code == 200
        detail = res_detail.json()
        assert len(detail["messages"]) == 2
        assert detail["title"] == "Self-attention là gì?"

    finally:
        client.delete(f"/api/conversations/{conv_id}")

