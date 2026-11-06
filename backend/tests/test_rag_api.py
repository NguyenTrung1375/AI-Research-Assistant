import uuid
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.rag.generation.rag_orchestrator import get_rag_orchestrator
from app.schemas.rag import RAGAnswer
from app.schemas.retrieval import Citation, RetrievedChunk

client = TestClient(app)


def test_api_rag_ask_returns_answer(monkeypatch):
    doc_id = uuid.uuid4()
    mock_orchestrator = MagicMock()
    mock_orchestrator.answer_question.return_value = RAGAnswer(
        query="Transformer là gì?",
        answer="Transformer là kiến trúc mạng nơ-ron sử dụng cơ chế self-attention [Trang 1].",
        citations=[
            Citation(
                document_id=doc_id,
                page_number=1,
                chunk_index=0,
                score=0.95,
                snippet="Transformer là kiến trúc mạng nơ-ron...",
            )
        ],
        context_chunks=[
            RetrievedChunk(
                id="c1",
                text="Transformer là kiến trúc mạng nơ-ron...",
                document_id=doc_id,
                page_number=1,
                chunk_index=0,
                score=0.95,
            )
        ],
        is_grounded=True,
    )

    # Override dependency
    app.dependency_overrides = {}
    monkeypatch.setattr(
        "app.api.routes.rag.get_rag_orchestrator",
        lambda: mock_orchestrator,
    )

    response = client.post(
        "/api/rag/ask",
        json={
            "query": "Transformer là gì?",
            "document_id": str(doc_id),
            "top_k": 3,
            "min_score": 0.4,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Transformer là gì?"
    assert data["is_grounded"] is True
    assert "Transformer là kiến trúc" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["page_number"] == 1


def test_api_rag_ask_validation_error_on_empty_query():
    response = client.post(
        "/api/rag/ask",
        json={"query": ""},
    )
    assert response.status_code == 422

