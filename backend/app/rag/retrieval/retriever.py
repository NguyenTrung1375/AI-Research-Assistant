import logging
import uuid
from functools import lru_cache
from typing import Any

from app.rag.embedding.embedding_service import EmbeddingService, get_embedding_service
from app.rag.vector_store.chroma_store import ChromaVectorStore, get_vector_store
from app.schemas.retrieval import Citation, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hệ thống truy xuất ngữ nghĩa (Retrieval System) cho RAG.
    Đảm nhận:
      - Query Embedding (Task 4.1)
      - Semantic Search & Top-K Retrieval (Task 4.2 & 4.3)
      - Metadata Filtering theo 1 hoặc nhiều tài liệu (Task 4.4 & 4.5)
      - Context Builder & Citation Extractor cho LLM (Task 4.6)
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    @property
    def vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def build_where_filter(
        self,
        document_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
    ) -> dict[str, Any] | None:
        """Xây dựng bộ lọc metadata (Task 4.4 & 4.5)."""
        if document_id:
            return {"document_id": str(document_id)}

        if document_ids:
            clean_ids = [str(d) for d in document_ids if d]
            if len(clean_ids) == 1:
                return {"document_id": clean_ids[0]}
            if len(clean_ids) > 1:
                return {"document_id": {"$in": clean_ids}}

        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Thực hiện truy xuất Top-K đoạn văn bản liên quan nhất dựa trên độ tương đồng ngữ nghĩa.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # 1. Tạo Query Embedding (Task 4.1)
        query_vector = self.embedding_service.embed_query(cleaned_query)

        # 2. Xây dựng Metadata Filter (Task 4.4 & 4.5)
        where_filter = self.build_where_filter(
            document_id=document_id,
            document_ids=document_ids,
        )

        # 3. Semantic Search trong ChromaDB (Task 4.2 & 4.3)
        raw_results = self.vector_store.query(
            query_embedding=query_vector,
            top_k=top_k,
            where=where_filter,
        )

        # 4. Lọc theo ngưỡng tương đồng tối thiểu (min_score)
        chunks: list[RetrievedChunk] = []
        for item in raw_results:
            score = item.get("score", 0.0)
            if score >= min_score:
                try:
                    doc_uuid = uuid.UUID(str(item.get("document_id")))
                except (ValueError, TypeError):
                    doc_uuid = uuid.uuid4()

                chunks.append(
                    RetrievedChunk(
                        id=str(item.get("id")),
                        text=str(item.get("text")),
                        document_id=doc_uuid,
                        page_number=int(item.get("page_number", 1)),
                        chunk_index=int(item.get("chunk_index", 0)),
                        score=score,
                        distance=item.get("distance"),
                    )
                )

        logger.info(
            "Retrieved %d relevant chunks for query '%s' (top_k=%d, min_score=%.2f)",
            len(chunks),
            cleaned_query,
            top_k,
            min_score,
        )
        return chunks

    @staticmethod
    def build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
        """
        Task 4.6 - Context Builder: Định dạng các đoạn văn bản truy xuất được thành
        khối context có cấu trúc chuẩn để nhúng vào prompt của LLM (Gemini).
        """
        if not retrieved_chunks:
            return "Không tìm thấy nội dung liên quan trong tài liệu."

        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            block = (
                f"--- NGUỒN [{idx}] (Tài liệu ID: {chunk.document_id} | Trang: {chunk.page_number} | Độ liên quan: {chunk.score:.2f}) ---\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    @staticmethod
    def build_citations(retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
        """Tạo danh sách trích dẫn nguồn có số trang phục vụ trả về cho Frontend."""
        citations = []
        for chunk in retrieved_chunks:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."

            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    snippet=snippet,
                )
            )
        return citations


@lru_cache
def get_retriever() -> Retriever:
    """Trả về thể hiện singleton của Retriever."""
    return Retriever()

