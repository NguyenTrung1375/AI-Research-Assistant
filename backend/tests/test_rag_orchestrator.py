import uuid
from unittest.mock import MagicMock
import pytest

from app.rag.generation.rag_orchestrator import RAGOrchestrator
from app.rag.retrieval.retriever import Retriever
from app.schemas.retrieval import RetrievedChunk


def test_rag_orchestrator_answers_grounded_question_with_citations():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    doc_id = uuid.uuid4()
    sample_chunk = RetrievedChunk(
        id="c1",
        text="Mô hình transformer loại bỏ hoàn toàn mạng tái tổ hợp (RNN) và tích chập (CNN).",
        document_id=doc_id,
        page_number=3,
        chunk_index=0,
        score=0.92,
    )
    mock_retriever.retrieve.return_value = [sample_chunk]
    mock_retriever.build_context.return_value = "Context with transformer info"
    mock_retriever.build_citations.return_value = [
        Retriever.build_citations([sample_chunk])[0]
    ]
    mock_llm.generate.return_value = "Transformer loại bỏ RNN và CNN [Trang 3]."

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    query = "Transformer có dùng RNN không?"
    result = orchestrator.answer_question(query=query, document_id=doc_id)

    # 1. Kiểm tra kết quả trả về
    assert result.is_grounded is True
    assert "Transformer loại bỏ RNN" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 3
    assert result.citations[0].document_id == doc_id

    # 2. Kiểm tra các bước phối hợp
    mock_retriever.retrieve.assert_called_once_with(
        query=query,
        top_k=5,
        document_id=doc_id,
        document_ids=None,
        min_score=0.3,
    )
    mock_llm.generate.assert_called_once()


def test_rag_orchestrator_handles_out_of_scope_query_without_calling_llm():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Không tìm thấy chunk nào liên quan
    mock_retriever.retrieve.return_value = []

    orchestrator = RAGOrchestrator(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    result = orchestrator.answer_question(query="Công thức nấu phở bò?")

    # 1. Phải đánh dấu không có căn cứ
    assert result.is_grounded is False
    assert "không có thông tin" in result.answer.lower()
    assert result.citations == []

    # 2. Bắt buộc: KHÔNG gọi LLM để tránh tốn token và tránh ảo giác
    mock_llm.generate.assert_not_called()


def test_rag_orchestrator_rejects_empty_query():
    orchestrator = RAGOrchestrator(retriever=MagicMock(), llm_service=MagicMock())
    with pytest.raises(ValueError, match="Query cannot be empty"):
        orchestrator.answer_question(query="   ")
