import logging
import uuid
from functools import lru_cache
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Quản lý lưu trữ và thao tác với ChromaDB vector database.
    Hỗ trợ PersistentClient, cấu hình không gian Cosine và metadata filtering.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._client = client
        self._collection: Any | None = None

    @property
    def client(self) -> Any:
        """Lazy load PersistentClient của ChromaDB."""
        if self._client is None:
            import chromadb

            logger.info("Initializing ChromaDB PersistentClient at: %s", self.persist_directory)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    @property
    def collection(self) -> Any:
        """Lấy hoặc tạo mới collection với metric Cosine."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(
        self,
        document_id: str | uuid.UUID,
        chunks: list[dict],
        embedding_service: Any | None = None,
    ) -> int:
        """
        Tạo vector embedding cho các chunks và lưu trữ kèm metadata vào ChromaDB.
        Mỗi chunk có id: {document_id}_p{page_number}_c{chunk_index} để đảm bảo tính duy nhất.
        """
        if not chunks:
            return 0

        doc_id_str = str(document_id)

        if embedding_service is None:
            from app.rag.embedding.embedding_service import get_embedding_service

            embedding_service = get_embedding_service()

        texts = []
        ids = []
        metadatas = []

        for chunk in chunks:
            page_number = int(chunk["page_number"])
            chunk_index = int(chunk["chunk_index"])
            text = chunk["text"]

            chunk_id = f"{doc_id_str}_p{page_number}_c{chunk_index}"

            texts.append(text)
            ids.append(chunk_id)
            metadatas.append({
                "document_id": doc_id_str,
                "page_number": page_number,
                "chunk_index": chunk_index,
            })

        # Tạo vector embedding theo batch
        embeddings = embedding_service.embed_documents(texts)

        # Lưu trữ vào ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "Added %d chunks for document %s to Chroma collection '%s'",
            len(chunks),
            doc_id_str,
            self.collection_name,
        )
        return len(chunks)

    def delete_by_document_id(self, document_id: str | uuid.UUID) -> None:
        """
        Xóa toàn bộ vectors và chunks thuộc về một document_id cụ thể.
        Đảm bảo không để lại vector mồ côi khi người dùng xóa tài liệu.
        """
        doc_id_str = str(document_id)
        self.collection.delete(where={"document_id": doc_id_str})
        logger.info("Deleted chunks for document %s from ChromaDB", doc_id_str)

    def count(self) -> int:
        """Tổng số chunk hiện có trong collection."""
        return int(self.collection.count())

    def count_by_document_id(self, document_id: str | uuid.UUID) -> int:
        """Đếm số chunk của một tài liệu cụ thể."""
        doc_id_str = str(document_id)
        result = self.collection.get(
            where={"document_id": doc_id_str},
            include=[],
        )
        return len(result.get("ids", []))

    def get_chunks_by_document_id(self, document_id: str | uuid.UUID) -> list[dict]:
        """Lấy danh sách các chunk đã lưu của một tài liệu."""
        doc_id_str = str(document_id)
        result = self.collection.get(
            where={"document_id": doc_id_str},
            include=["documents", "metadatas"],
        )

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks = []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            chunks.append({
                "id": chunk_id,
                "text": text,
                "document_id": meta.get("document_id"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
            })

        chunks.sort(key=lambda x: (x["page_number"], x["chunk_index"]))
        return chunks

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Task 4.2 & 4.3: Tìm kiếm các chunk có độ tương đồng ngữ nghĩa cao nhất.
        Trả về danh sách các chunk kèm score độ tương đồng chuẩn hóa (0.0 - 1.0).
        """
        if self.count() == 0:
            return []

        actual_k = min(top_k, self.count())
        if actual_k <= 0:
            return []

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self.collection.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query error: %s", exc)
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # Với không gian Cosine và L2 normalized vector: similarity = 1 - distance
            similarity = 1.0 - float(dist)
            items.append({
                "id": chunk_id,
                "text": text,
                "document_id": meta.get("document_id"),
                "page_number": int(meta.get("page_number", 1)),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "distance": round(float(dist), 4),
                "score": round(max(0.0, min(1.0, similarity)), 4),
            })

        # Sắp xếp theo score giảm dần (độ liên quan cao nhất đứng đầu)
        items.sort(key=lambda x: x["score"], reverse=True)
        return items



@lru_cache
def get_vector_store() -> ChromaVectorStore:
    """Trả về thể hiện singleton của ChromaVectorStore."""
    return ChromaVectorStore()

