import numpy as np

from rag.embedding import QwenEmbeddingModel


model = QwenEmbeddingModel()

texts = [
    "FastAPI 使用 Depends 实现依赖注入。",
    "Dependency injection allows dependencies to be provided externally.",
    "香蕉是一种常见水果。",
]

embeddings = model.embed(texts)

a = np.array(embeddings[0])
b = np.array(embeddings[1])
c = np.array(embeddings[2])

print("FastAPI vs DI:", a @ b)
print("FastAPI vs Banana:", a @ c)