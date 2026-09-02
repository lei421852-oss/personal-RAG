from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.loader import load_markdown_documents
from rag.vector_store import ChromaVectorStore


def main() -> None:
    documents = load_markdown_documents("data/raw")

    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=500,
                chunk_overlap=100,
            )
        )

    embedding_model = QwenEmbeddingModel()
    embeddings = embedding_model.embed_documents(
        [chunk.text for chunk in chunks]
    )

    vector_store = ChromaVectorStore(rebuild=True)
    vector_store.add(chunks=chunks, embeddings=embeddings)

    print(f"documents loaded: {len(documents)}")
    print(f"chunks created: {len(chunks)}")
    print(f"vectors written: {len(chunks)}")


if __name__ == "__main__":
    main()
