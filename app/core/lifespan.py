import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    vector_store = ChromaVectorStore()
    embedding_model = QwenEmbeddingModel()
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    llm = DeepSeekLLMClient(api_key=os.environ["DEEPSEEK_API_KEY"])

    app.state.rag_service = RAGService(retriver=retriever, llm=llm)

    yield
