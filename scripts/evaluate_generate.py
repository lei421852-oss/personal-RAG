"""Run generation evaluation with the configured RAG application resources."""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from rag.embedding import QwenEmbeddingModel
from rag.llm import DeepSeekLLMClient
from rag.retriever import Retriever
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore

cases = json.loads(Path("data/eval/generation_eval.json").read_text(encoding="utf-8"))

settings = get_settings()

embedding_model = QwenEmbeddingModel(
    model_name=settings.embedding_model_name,
)

vector_store = ChromaVectorStore(
    persist_directory=settings.chroma_persist_directory,
    collection_name=settings.chroma_collection_name,
)

retriever = Retriever(embedding_model=embedding_model, vector_store=vector_store)

llm = DeepSeekLLMClient(
    api_key=settings.require_deepseek_api_key(),
    model=settings.deepseek_model,
)

rag = RAGService(retriver=retriever, llm=llm)

for index, case in enumerate(cases, start=1):
    result = rag.answer(
        case["question"],
        top_k=settings.default_top_k,
    )

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
    for i, source in enumerate(result.sources, start=1):
        print(
            f"[{i}] ",
            source.source,
            source.chunk_id,
            source.score,
        )
