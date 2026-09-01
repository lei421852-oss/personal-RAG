from dataclasses import dataclass

import numpy as np

from rag.document import Chunk
from rag.embedding import EmbeddingModel


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], embedding_model: EmbeddingModel):
        self.chunks = chunks
        self.embedding_model = embedding_model

        texts = [chunk.text for chunk in chunks]

        self.embeddings = np.array(embedding_model.embed_documents(texts))

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        query_embedding = np.array(self.embedding_model.embed_query(query=query))

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            for index in top_indices
        ]
