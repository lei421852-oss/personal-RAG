from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.loader import load_markdown_documents
from rag.prompt import build_rag_prompt
from rag.retriever import Retriever

documents = load_markdown_documents("data/raw")

chunks = []

for document in documents:
    chunks.extend(chunk_document(document=document, chunk_size=500, chunk_overlap=100))

embedding_model = QwenEmbeddingModel()

retriever = Retriever(chunks=chunks, embedding_model=embedding_model)

question = "Mini GPT 为什么使用 FastAPI lifespan？"

results = retriever.search(
    query=question,
    top_k=3,
)

prompt = build_rag_prompt(
    question=question,
    chunks=results,
)

print(prompt)
