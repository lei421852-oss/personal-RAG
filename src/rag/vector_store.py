from pathlib import Path

import chromadb

from rag.document import Chunk
from rag.retriever import RetrievedChunk


class IndexNotReadyError(RuntimeError):
    pass


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "personal-rag",
        *,
        rebuild: bool = False,
    ):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection_name = collection_name

        existing_names = {
            collection if isinstance(collection, str) else collection.name
            for collection in self.client.list_collections()
        }

        if rebuild:
            if collection_name in existing_names:
                self.client.delete_collection(name=collection_name)

            self.collection = self.client.create_collection(
                name=collection_name,
                configuration={"hnsw": {"space": "cosine"}},
            )
            return

        if collection_name not in existing_names:
            raise IndexNotReadyError(
                "Chroma index is not ready. Run `python scripts/build_index.py` "
                "before starting the application."
            )

        self.collection = self.client.get_collection(name=collection_name)

        if self.collection.count() == 0:
            raise IndexNotReadyError(
                "Chroma index is empty. Run `python scripts/build_index.py` "
                "before starting the application."
            )

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        if not chunks:
            return

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source, "chunk_id": chunk.chunk_id}
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        result_count = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            if document is None or metadata is None:
                continue

            chunk = Chunk(
                text=document,
                source=str(metadata["source"]),
                chunk_id=str(metadata["chunk_id"]),
            )
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0 - float(distance),
                )
            )

        return retrieved_chunks
