import math
import numpy as np
import pytest

from app.rag.embedding.embedding_service import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    EmbeddingService,
)


class DummySentenceTransformer:
    """Mock SentenceTransformer để kiểm thử embedding service nhanh chóng và chuẩn xác."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.last_encoded_inputs = None
        self.last_batch_size = None
        self.last_normalize_flag = None

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        self.last_encoded_inputs = sentences
        self.last_batch_size = batch_size
        self.last_normalize_flag = normalize_embeddings

        if isinstance(sentences, str):
            vec = np.full(self.dim, 1.0, dtype=np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            return vec

        results = []
        for _ in sentences:
            vec = np.full(self.dim, 1.0, dtype=np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return np.array(results)


def test_embed_documents_adds_passage_prefix_and_normalizes():
    dummy_model = DummySentenceTransformer(dim=768)
    service = EmbeddingService(model=dummy_model)

    documents = [
        "Mô hình RAG giúp giảm ảo giác (hallucination) trong LLM.",
        "PostgreSQL và ChromaDB kết hợp quản lý metadata và vector.",
    ]

    embeddings = service.embed_documents(documents, batch_size=16)

    # 1. Kiểm tra tiền tố passage: theo chuẩn E5
    assert dummy_model.last_encoded_inputs == [
        f"{PASSAGE_PREFIX}{documents[0]}",
        f"{PASSAGE_PREFIX}{documents[1]}",
    ]
    assert dummy_model.last_batch_size == 16
    assert dummy_model.last_normalize_flag is True

    # 2. Kiểm tra kích thước và kiểu dữ liệu
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    assert len(embeddings[1]) == 768

    # 3. Kiểm tra L2 norm xấp xỉ 1.0
    norm_0 = math.sqrt(sum(x * x for x in embeddings[0]))
    norm_1 = math.sqrt(sum(x * x for x in embeddings[1]))
    assert math.isclose(norm_0, 1.0, rel_tol=1e-5)
    assert math.isclose(norm_1, 1.0, rel_tol=1e-5)


def test_embed_query_adds_query_prefix():
    dummy_model = DummySentenceTransformer(dim=768)
    service = EmbeddingService(model=dummy_model)

    query = "Hệ thống RAG hoạt động như thế nào?"
    embedding = service.embed_query(query)

    # 1. Kiểm tra tiền tố query: theo chuẩn E5
    assert dummy_model.last_encoded_inputs == f"{QUERY_PREFIX}{query}"

    # 2. Kiểm tra định dạng vector
    assert len(embedding) == 768
    norm = math.sqrt(sum(x * x for x in embedding))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_embed_documents_empty_list_returns_empty():
    dummy_model = DummySentenceTransformer(dim=768)
    service = EmbeddingService(model=dummy_model)

    assert service.embed_documents([]) == []


def test_embed_documents_invalid_type_raises_type_error():
    dummy_model = DummySentenceTransformer(dim=768)
    service = EmbeddingService(model=dummy_model)

    with pytest.raises(TypeError, match="must be a string"):
        service.embed_documents(["hợp lệ", 12345])  # type: ignore


def test_embed_query_invalid_or_empty_raises_error():
    dummy_model = DummySentenceTransformer(dim=768)
    service = EmbeddingService(model=dummy_model)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.embed_query("   ")

    with pytest.raises(TypeError, match="Query must be a string"):
        service.embed_query(None)  # type: ignore


def test_compute_similarity_cosine():
    # 2 vector giống hệt nhau -> similarity = 1.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert math.isclose(EmbeddingService.compute_similarity(vec1, vec2), 1.0)

    # 2 vector trực giao -> similarity = 0.0
    vec3 = [0.0, 1.0, 0.0]
    assert math.isclose(EmbeddingService.compute_similarity(vec1, vec3), 0.0)


def test_embedding_service_dimension_property():
    dummy_model = DummySentenceTransformer(dim=512)
    service = EmbeddingService(model=dummy_model)
    assert service.dimension == 512

