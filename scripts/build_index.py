# ruff: noqa: E402

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging
from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.loader import load_markdown_documents
from rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    logger.info("index build started")

    try:
        build_index()
    except Exception:
        logger.exception("index build failed")
        raise

    logger.info("index build completed")


def build_index() -> None:
    settings = get_settings()
    documents = load_markdown_documents(settings.raw_data_dir)
    logger.info("documents loaded count=%d", len(documents))

    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    logger.info("chunks created count=%d", len(chunks))

    logger.info("embedding started model=%s", settings.embedding_model_name)
    embedding_model = QwenEmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])
    logger.info("embedding completed vector_count=%d", len(embeddings))

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        rebuild=True,
    )
    logger.info(
        "collection rebuilt collection=%s",
        settings.chroma_collection_name,
    )
    vector_store.add(chunks=chunks, embeddings=embeddings)
    logger.info("vectors written count=%d", len(chunks))


if __name__ == "__main__":
    main()
