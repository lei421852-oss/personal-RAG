from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    deepseek_api_key = settings.require_deepseek_api_key()

    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    embedding_model = QwenEmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    llm = DeepSeekLLMClient(
        api_key=deepseek_api_key,
        model=settings.deepseek_model,
    )

    app.state.rag_service = RAGService(retriver=retriever, llm=llm)

    yield
