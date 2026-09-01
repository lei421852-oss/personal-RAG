import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.loader import load_markdown_documents
from rag.retriever import Retriever
from rag.service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    retriever = Retriever(chunks=chunks, embedding_model=embedding_model)
    llm = DeepSeekLLMClient(api_key=os.environ["DEEPSEEK_API_KEY"])

    app.state.rag_service = RAGService(retriver=retriever, llm=llm)

    yield
