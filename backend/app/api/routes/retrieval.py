from fastapi import APIRouter, status

from app.rag.retrieval.retriever import get_retriever
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse

router = APIRouter(prefix="/api/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
def search_documents(
    payload: RetrievalRequest,
) -> RetrievalResponse:
    """
    Task 4.2 - 4.6: Truy xuất ngữ nghĩa (Semantic Search), lọc metadata và tạo context RAG.
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
        document_ids=payload.document_ids,
        min_score=payload.min_score,
    )
    context_text = retriever.build_context(chunks)
    citations = retriever.build_citations(chunks)

    return RetrievalResponse(
        query=payload.query,
        total_retrieved=len(chunks),
        chunks=chunks,
        citations=citations,
        context_text=context_text,
    )

