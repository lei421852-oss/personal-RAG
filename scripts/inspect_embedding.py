from rag.embedding import QwenEmbeddingModel

model = QwenEmbeddingModel()

texts = [
    "FastAPI 使用 Depends 实现依赖注入。",
    "Dependency injection allows dependencies to be provided externally.",
    "香蕉是一种水果。",
]

embeddings = model._embed(texts)

print("num embeddings:", len(embeddings))
print("dimension:", len(embeddings[0]))
print("first values:", embeddings[0][:5])
