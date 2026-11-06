import logging
import uuid
from functools import lru_cache

from app.rag.generation.llm_service import LLMService, get_llm_service
from app.rag.generation.prompt_template import (
    build_rag_user_prompt,
    build_system_instruction,
)
from app.rag.retrieval.retriever import Retriever, get_retriever
from app.schemas.rag import RAGAnswer

logger = logging.getLogger(__name__)

FALLBACK_OUT_OF_SCOPE_MESSAGE = (
    "Tài liệu được cung cấp không có thông tin hoặc không đủ căn cứ để trả lời câu hỏi này."
)


class RAGOrchestrator:
    """
    Deep Module điều phối toàn bộ luồng RAG:
      - Semantic Retrieval (Retriever)
      - Lọc câu hỏi ngoài phạm vi (Out-of-Scope Detection)
      - Bơm Context & System Prompt chống ảo giác
      - Sinh câu trả lời bám sát sự thật qua Google Gemini (LLMService)
      - Đính kèm trích dẫn nguồn có số trang chính xác (Citations)
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_service: LLMService | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm_service = llm_service

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service

    def answer_question(
        self,
        query: str,
        document_id: uuid.UUID | None = None,
        document_ids: list[uuid.UUID] | None = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> RAGAnswer:
        """
        Hàm trung tâm giải đáp câu hỏi của người dùng dựa trên tài liệu.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        # 1. Truy xuất các đoạn văn bản liên quan nhất (Task 4.1 - 4.5)
        chunks = self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            document_id=document_id,
            document_ids=document_ids,
            min_score=min_score,
        )

        # 2. Xử lý Out-of-Scope (Task 5.9): Nếu không có thông tin, không gọi LLM để tiết kiệm token
        if not chunks:
            logger.info("No relevant chunks found for query: '%s'. Returning fallback response.", cleaned_query)
            return RAGAnswer(
                query=cleaned_query,
                answer=FALLBACK_OUT_OF_SCOPE_MESSAGE,
                citations=[],
                context_chunks=[],
                is_grounded=False,
            )

        # 3. Bơm Context và cấu trúc hóa Prompt (Task 4.6 & Task 5.2, 5.5)
        context_text = self.retriever.build_context(chunks)
        user_prompt = build_rag_user_prompt(query=cleaned_query, context=context_text)
        system_instruction = build_system_instruction()

        # 4. Sinh câu trả lời có căn cứ từ Google Gemini (Task 5.6)
        llm_answer = self.llm_service.generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.2,
        )

        # 5. Gắn danh sách trích dẫn nguồn có số trang cụ thể (Task 5.7 & 5.8)
        citations = self.retriever.build_citations(chunks)

        return RAGAnswer(
            query=cleaned_query,
            answer=llm_answer,
            citations=citations,
            context_chunks=chunks,
            is_grounded=True,
        )


@lru_cache
def get_rag_orchestrator() -> RAGOrchestrator:
    """Trả về thể hiện singleton của RAGOrchestrator."""
    return RAGOrchestrator()

