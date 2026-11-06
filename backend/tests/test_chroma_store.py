import uuid
import numpy as np
import pytest

from app.rag.embedding.embedding_service import EmbeddingService
from app.rag.vector_store.chroma_store import ChromaVectorStore


class MockEmbeddingService:
    """Mock EmbeddingService để test ChromaVectorStore độc lập và nhanh chóng."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for idx, _ in enumerate(texts):
            # Tạo vector đơn vị có chút biến thiên để phân biệt
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[idx % self.dim] = 1.0
            results.append(vec.tolist())
        return results

    def embed_query(self, query: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = 1.0
        return vec.tolist()


def test_add_chunks_stores_vectors_and_metadata(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )
    mock_embedder = MockEmbeddingService(dim=768)
    doc_id = uuid.uuid4()

    chunks = [
        {"page_number": 1, "chunk_index": 0, "text": "Đoạn 1 trang 1 về RAG."},
        {"page_number": 1, "chunk_index": 1, "text": "Đoạn 2 trang 1 về Vector Store."},
        {"page_number": 2, "chunk_index": 0, "text": "Đoạn 1 trang 2 về LLM Gemini."},
    ]

    added_count = store.add_chunks(
        document_id=doc_id,
        chunks=chunks,
        embedding_service=mock_embedder,
    )

    assert added_count == 3
    assert store.count() == 3
    assert store.count_by_document_id(doc_id) == 3

    saved_chunks = store.get_chunks_by_document_id(doc_id)
    assert len(saved_chunks) == 3

    # Kiểm tra cấu trúc ID và metadata
    assert saved_chunks[0]["id"] == f"{doc_id}_p1_c0"
    assert saved_chunks[0]["page_number"] == 1
    assert saved_chunks[0]["chunk_index"] == 0
    assert saved_chunks[0]["text"] == "Đoạn 1 trang 1 về RAG."

    assert saved_chunks[2]["id"] == f"{doc_id}_p2_c0"
    assert saved_chunks[2]["page_number"] == 2
    assert saved_chunks[2]["chunk_index"] == 0


def test_delete_by_document_id_removes_only_target_vectors(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_isolation"),
        collection_name="test_isolation",
    )
    mock_embedder = MockEmbeddingService(dim=768)

    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    chunks_a = [
        {"page_number": 1, "chunk_index": 0, "text": "Tài liệu A - chunk 1."},
        {"page_number": 1, "chunk_index": 1, "text": "Tài liệu A - chunk 2."},
    ]
    chunks_b = [
        {"page_number": 1, "chunk_index": 0, "text": "Tài liệu B - chunk 1."},
    ]

    store.add_chunks(doc_a_id, chunks_a, mock_embedder)
    store.add_chunks(doc_b_id, chunks_b, mock_embedder)

    assert store.count() == 3
    assert store.count_by_document_id(doc_a_id) == 2
    assert store.count_by_document_id(doc_b_id) == 1

    # Xóa tài liệu A
    store.delete_by_document_id(doc_a_id)

    # Tài liệu A phải về 0, tài liệu B phải giữ nguyên
    assert store.count_by_document_id(doc_a_id) == 0
    assert store.count_by_document_id(doc_b_id) == 1
    assert store.count() == 1


def test_add_empty_chunks_returns_zero(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_empty"),
        collection_name="test_empty",
    )
    mock_embedder = MockEmbeddingService(dim=768)

    res = store.add_chunks(uuid.uuid4(), [], mock_embedder)
    assert res == 0
    assert store.count() == 0

