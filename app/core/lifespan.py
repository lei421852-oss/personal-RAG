import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import configure_logging
from rag.embedding import QwenEmbeddingModel
from rag.errors import IndexNotReadyError
from rag.llm import DeepSeekLLMClient
from rag.reranker import BGEReranker
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("RAG application startup started")
    settings = get_settings()
    deepseek_api_key = settings.require_deepseek_api_key()

    logger.info(
        "opening Chroma vector store collection=%s",
        settings.chroma_collection_name,
    )
    try:
        vector_store = ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        )
    except IndexNotReadyError:
        logger.exception(
            "Chroma index is not ready; run python scripts/build_index.py first"
        )
        raise
    except Exception:
        logger.exception("failed to open Chroma vector store")
        raise
    logger.info(
        "Chroma collection ready collection=%s vector_count=%d",
        settings.chroma_collection_name,
        vector_store.collection.count(),
    )

    logger.info("loading embedding model model=%s", settings.embedding_model_name)
    embedding_model = QwenEmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    logger.info("embedding model ready model=%s", settings.embedding_model_name)

    logger.info("creating retriever")
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    logger.info("loading reranker model model=%s", settings.reranker_model_name)
    reranker = BGEReranker(model_name=settings.reranker_model_name)
    logger.info("reranker model ready model=%s", settings.reranker_model_name)

    logger.info("creating LLM client model=%s", settings.deepseek_model)
    llm = DeepSeekLLMClient(
        api_key=deepseek_api_key,
        model=settings.deepseek_model,
    )

    app.state.rag_service = RAGService(
        retriver=retriever,
        reranker=reranker,
        llm=llm,
    )
    logger.info("RAG service ready")
    logger.info("RAG application startup completed")

    yield
