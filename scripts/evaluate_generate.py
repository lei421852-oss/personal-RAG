import json
from pathlib import Path

from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore

import os

cases = json.loads(
    Path("data/eval/generation_eval.json").read_text(encoding="utf-8")
)

embedding_model = QwenEmbeddingModel()

vector_store = ChromaVectorStore(
    persist_directory="data/chroma",
    collection_name="personal-rag"
)

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

llm = DeepSeekLLMClient(
    api_key="sk-2eaf292e7fe14a21a010815d31167e3e"
)

rag = RAGService(
    retriver=retriever,
    llm=llm
)

for index,case in enumerate(cases, start=1):
    result = rag.answer(case["question"])

    print("\n" + "=" * 60)
    print(f"Case {index}")
    print("=" * 60)

    print("\nQuestion:")
    print(case["question"])

    print("\nExpected:")
    print(case.get("expected_answer"))

    print("\nActual:")
    print(result.answer)

    print("\nSources:")
    for i,source in enumerate(result.sources,start=1):
        print(
            f"[{i}] ",
            source.source,
            source.chunk_id,
            source.score,
        )