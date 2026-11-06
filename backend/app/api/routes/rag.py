from fastapi import APIRouter, status

from app.rag.generation.rag_orchestrator import get_rag_orchestrator
from app.schemas.rag import RAGAnswer, RAGQuestionRequest

router = APIRouter(prefix="/api/rag", tags=["RAG"])


@router.post(
    "/ask",
    response_model=RAGAnswer,
    status_code=status.HTTP_200_OK,
)
def ask_question(
    payload: RAGQuestionRequest,
) -> RAGAnswer:
    """
    Task 5.4 - 5.9: Hỏi đáp tài liệu nghiên cứu thông qua RAG Pipeline.
    Tự động truy xuất ngữ cảnh, lọc phạm vi tài liệu, chống ảo giác và trích dẫn số trang.
    """
    orchestrator = get_rag_orchestrator()
    return orchestrator.answer_question(
        query=payload.query,
        document_id=payload.document_id,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        min_score=payload.min_score,
    )

