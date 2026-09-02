# ruff: noqa: E402

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.loader import load_markdown_documents
from rag.vector_store import ChromaVectorStore


def main() -> None:
    settings = get_settings()
    documents = load_markdown_documents(settings.raw_data_dir)

    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    embedding_model = QwenEmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        rebuild=True,
    )
    vector_store.add(chunks=chunks, embeddings=embeddings)

    print(f"documents loaded: {len(documents)}")
    print(f"chunks created: {len(chunks)}")
    print(f"vectors written: {len(chunks)}")


if __name__ == "__main__":
    main()
