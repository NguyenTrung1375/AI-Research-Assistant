import logging
from functools import lru_cache
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class EmbeddingService:
    """
    Dịch vụ tạo vector embedding cho văn bản và câu truy vấn.
    Sử dụng mô hình multilingual-e5-base theo chuẩn tiền tố:
      - 'passage: ' cho tài liệu (documents / chunks)
      - 'query: ' cho câu hỏi người dùng (retrieval queries)
    """

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        """Lazy load mô hình để tối ưu thời gian khởi động và tài nguyên bộ nhớ."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Kích thước vector đầu ra của mô hình (768 chiều với multilingual-e5-base)."""
        if self._dimension is None:
            if hasattr(self.model, "get_embedding_dimension"):
                self._dimension = self.model.get_embedding_dimension()
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self._dimension = self.model.get_sentence_embedding_dimension()
            else:
                self._dimension = 768
        return self._dimension

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Tạo vector embedding cho danh sách các đoạn văn bản (chunks).
        Tự động thêm tiền tố 'passage: ' theo quy định của mô hình E5.
        Tất cả vector được chuẩn hóa L2 (norm = 1.0) cho phép tính Cosine Similarity nhanh chóng.
        """
        if not texts:
            return []

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Text at index {idx} must be a string, got {type(text).__name__}.")

        prefixed_texts = [f"{PASSAGE_PREFIX}{t}" for t in texts]

        raw_embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(raw_embeddings, np.ndarray):
            return raw_embeddings.tolist()
        return [list(vec) for vec in raw_embeddings]

    def embed_query(self, query: str) -> list[float]:
        """
        Tạo vector embedding cho một câu truy vấn tìm kiếm (search query).
        Tự động thêm tiền tố 'query: ' theo quy định của mô hình E5.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")

        prefixed_query = f"{QUERY_PREFIX}{cleaned_query}"

        raw_embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(raw_embedding, np.ndarray):
            return raw_embedding.tolist()
        return list(raw_embedding)

    @staticmethod
    def compute_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """
        Tính toán Cosine Similarity giữa 2 vector đã được L2-normalized.
        Với 2 vector có norm = 1.0, Cosine Similarity chính bằng Dot Product.
        """
        a = np.array(vector_a, dtype=np.float32)
        b = np.array(vector_b, dtype=np.float32)
        return float(np.dot(a, b))


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Trả về thể hiện singleton của EmbeddingService."""
    return EmbeddingService()

