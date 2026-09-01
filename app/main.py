import os

from fastapi import FastAPI

from app.api.schemas import QueryRequest, QueryResponse
from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.loader import load_markdown_documents
from rag.retriever import Retriever
from rag.service import RAGService

app = FastAPI()

documents = load_markdown_documents("data/raw")

chunks = []

for document in documents:
    chunks.extend(chunk_document(document=document, chunk_size=500, chunk_overlap=100))

embedding_model = QwenEmbeddingModel()

retriever = Retriever(chunks=chunks, embedding_model=embedding_model)

llm = DeepSeekLLMClient(api_key="sk-2eaf292e7fe14a21a010815d31167e3e")

rag_service = RAGService(retriver=retriever, llm=llm)


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest):
    result = rag_service.answer(question=request.question, top_k=request.top_k)

    return result
