from rag.chunker import chunk_document
from rag.embedding import QwenEmbeddingModel
from rag.loader import load_markdown_documents
from rag.retriever import Retriever

documents = load_markdown_documents("data/raw")

chunks = []

for document in documents:
    chunks.extend(
        chunk_document(
            document=document,
            chunk_size=500,
            chunk_overlap=100
        )
    )

embedding_model = QwenEmbeddingModel()

retriever = Retriever(
    chunks=chunks,
    embedding_model=embedding_model
)

query = "Mini Gpt 为什么要用FastAPI Lifispan"

results = retriever.search(
    query=query,
    top_k=3
)

for result in results:
    print("score:",result.score)
    print("source:",result.chunk.source)
    print("chunk_id:",result.chunk.chunk_id)
    print(result.chunk.text)
    print("="*100)