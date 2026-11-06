import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.embedding.embedding_service import EmbeddingService
from app.rag.retrieval.retriever import Retriever
from app.rag.vector_store.chroma_store import ChromaVectorStore
from app.schemas.retrieval import RetrievedChunk

client = TestClient(app)


class MockEmbeddingService:
    """Mock để kiểm thử retrieval nhanh và xác định trước độ tương đồng."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            t = text.lower()
            if "deep learning" in t or "transformer" in t or "nlp" in t:
                # Vector hướng về trục 0
                results.append([1.0, 0.0, 0.0, 0.0])
            elif "machine learning" in t:
                # Vector gần trục 0
                results.append([0.7071, 0.7071, 0.0, 0.0])
            elif "cooking" in t or "bread" in t:
                # Vector trực giao (khác chủ đề)
                results.append([0.0, 0.0, 1.0, 0.0])
            else:
                results.append([0.0, 1.0, 0.0, 0.0])
        return results

    def embed_query(self, query: str) -> list[float]:
        q = query.lower()
        if "transformer" in q or "nlp" in q:
            return [1.0, 0.0, 0.0, 0.0]
        if "cooking" in q:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.5, 0.5]


def test_retriever_semantic_ordering(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_retrieval"),
        collection_name="test_retrieval",
    )
    mock_embed = MockEmbeddingService(dim=4)
    retriever = Retriever(vector_store=store, embedding_service=mock_embed)

    doc_id = uuid.uuid4()
    chunks = [
        {"page_number": 1, "chunk_index": 0, "text": "Deep learning and transformer models in NLP."},
        {"page_number": 2, "chunk_index": 0, "text": "Basic machine learning models."},
        {"page_number": 3, "chunk_index": 0, "text": "Cooking delicious sourdough bread."},
    ]

    store.add_chunks(doc_id, chunks, mock_embed)

    results = retriever.retrieve(
        query="transformers and NLP",
        top_k=3,
    )

    assert len(results) == 3
    # Đoạn về transformer phải có score cao nhất (score = 1.0)
    assert "transformer" in results[0].text.lower()
    assert results[0].score >= results[1].score
    assert results[1].score >= results[2].score


def test_retriever_metadata_filter_single_document(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_filter_single"),
        collection_name="test_filter_single",
    )
    mock_embed = MockEmbeddingService(dim=4)
    retriever = Retriever(vector_store=store, embedding_service=mock_embed)

    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    store.add_chunks(doc_a, [{"page_number": 1, "chunk_index": 0, "text": "Transformer paper A."}], mock_embed)
    store.add_chunks(doc_b, [{"page_number": 1, "chunk_index": 0, "text": "Transformer paper B."}], mock_embed)

    # Lọc chỉ theo doc_a
    results = retriever.retrieve(query="transformer", top_k=5, document_id=doc_a)
    assert len(results) == 1
    assert results[0].document_id == doc_a


def test_retriever_metadata_filter_multiple_documents(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_filter_multi"),
        collection_name="test_filter_multi",
    )
    mock_embed = MockEmbeddingService(dim=4)
    retriever = Retriever(vector_store=store, embedding_service=mock_embed)

    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    doc_c = uuid.uuid4()

    store.add_chunks(doc_a, [{"page_number": 1, "chunk_index": 0, "text": "Doc A content."}], mock_embed)
    store.add_chunks(doc_b, [{"page_number": 1, "chunk_index": 0, "text": "Doc B content."}], mock_embed)
    store.add_chunks(doc_c, [{"page_number": 1, "chunk_index": 0, "text": "Doc C content."}], mock_embed)

    # Lọc chỉ theo doc_a và doc_b
    results = retriever.retrieve(query="content", top_k=5, document_ids=[doc_a, doc_b])
    retrieved_doc_ids = {r.document_id for r in results}
    assert doc_c not in retrieved_doc_ids
    assert doc_a in retrieved_doc_ids or doc_b in retrieved_doc_ids


def test_retriever_min_score_filtering(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_score_filter"),
        collection_name="test_score_filter",
    )
    mock_embed = MockEmbeddingService(dim=4)
    retriever = Retriever(vector_store=store, embedding_service=mock_embed)

    doc_id = uuid.uuid4()
    chunks = [
        {"page_number": 1, "chunk_index": 0, "text": "Deep learning transformers."},
        {"page_number": 2, "chunk_index": 0, "text": "Cooking food bread."},
    ]
    store.add_chunks(doc_id, chunks, mock_embed)

    # Đặt ngưỡng min_score = 0.8 -> Đoạn cooking (score = 0.0) phải bị loại bỏ
    results = retriever.retrieve(
        query="transformers in deep learning",
        min_score=0.8,
    )
    assert len(results) == 1
    assert "transformers" in results[0].text.lower()


def test_build_context_and_citations():
    doc_id = uuid.uuid4()
    chunks = [
        RetrievedChunk(
            id="c1",
            text="Hệ thống RAG sử dụng vector search để tìm kiếm tri thức.",
            document_id=doc_id,
            page_number=3,
            chunk_index=1,
            score=0.92,
        ),
        RetrievedChunk(
            id="c2",
            text="ChromaDB hỗ trợ cosine distance và metadata filtering.",
            document_id=doc_id,
            page_number=5,
            chunk_index=0,
            score=0.88,
        ),
    ]

    # Kiểm tra context builder
    context = Retriever.build_context(chunks)
    assert "NGUỒN [1]" in context
    assert "Trang: 3" in context
    assert "Độ liên quan: 0.92" in context
    assert "NGUỒN [2]" in context
    assert "Trang: 5" in context

    # Kiểm tra citations
    citations = Retriever.build_citations(chunks)
    assert len(citations) == 2
    assert citations[0].page_number == 3
    assert citations[0].score == 0.92
    assert "vector search" in citations[0].snippet


def test_api_retrieval_search_endpoint():
    response = client.post(
        "/api/retrieval/search",
        json={
            "query": "nghiên cứu về trí tuệ nhân tạo",
            "top_k": 3,
            "min_score": 0.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "total_retrieved" in data
    assert "chunks" in data
    assert "citations" in data
    assert "context_text" in data

