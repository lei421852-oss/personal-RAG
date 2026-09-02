"""Run retrieval evaluation against the configured persistent index."""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from rag.embedding import QwenEmbeddingModel
from rag.retriever import Retriever
from rag.vector_store import ChromaVectorStore


def is_hit(result, expected_source: str, expected_text: str) -> bool:
    return result.chunk.source == expected_source and expected_text in result.chunk.text


eval_path = Path("data/eval/retriever_eval.json")

cases = json.loads(eval_path.read_text(encoding="utf-8"))

settings = get_settings()

embedding_model = QwenEmbeddingModel(
    model_name=settings.embedding_model_name,
)

vector_store = ChromaVectorStore(
    persist_directory=settings.chroma_persist_directory,
    collection_name=settings.chroma_collection_name,
)

retriever = Retriever(embedding_model=embedding_model, vector_store=vector_store)

hit_counts = {1: 0, 3: 0, 5: 0}

for case in cases:
    question = case["question"]
    expected_source = case["expected_source"]
    expected_text = case["expected_text"]

    results = retriever.search(query=question, top_k=5)

    for k in hit_counts:
        tok_k_result = results[:k]

        if any(is_hit(result, expected_source, expected_text) for result in results):
            hit_counts[k] += 1

    if not any(is_hit(result, expected_source, expected_text) for result in results):
        print("\nMISS")
        print("question:", question)
        print("expected source:", expected_source)
        print("expected text:", expected_text)

        for rank, result in enumerate(results, start=1):
            print(
                rank,
                "|",
                result.chunk.source,
                "|",
                result.chunk.chunk_id,
                "|",
                result.score,
                "|",
                result.chunk.text,
            )

total = len(cases)

print("\nRetriever Evaluation")
print("-" * 50)

for k, hits in hit_counts.items():
    print(f"Hit@{k}:{hits / total:.3f}({hits}/{total})")
