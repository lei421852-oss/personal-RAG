from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.document import Chunk
from rag.embedding import EmbeddingModel

if TYPE_CHECKING:
    from rag.vector_store import ChromaVectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: "ChromaVectorStore",
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.embed_query(query=query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
