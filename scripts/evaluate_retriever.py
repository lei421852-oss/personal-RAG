"""Run retrieval evaluation against the configured persistent index."""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from rag.embedding import QwenEmbeddingModel
from rag.reranker import BGEReranker
from rag.retriever import RetrievedChunk, Retriever
from rag.vector_store import ChromaVectorStore


def is_hit(
    result: RetrievedChunk,
    expected_source: str,
    expected_text: str,
) -> bool:
    return result.chunk.source == expected_source and expected_text in result.chunk.text


def find_rank(
    results: list[RetrievedChunk],
    expected_source: str,
    expected_text: str,
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if is_hit(result, expected_source, expected_text):
            return rank

    return None


def format_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "not found"


def main() -> None:
    cases = json.loads(
        Path("data/eval/retriever_eval.json").read_text(encoding="utf-8")
    )
    settings = get_settings()
    embedding_model = QwenEmbeddingModel(
        model_name=settings.embedding_model_name,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    reranker = BGEReranker(model_name=settings.reranker_model_name)

    before_hits = {1: 0, 3: 0}
    after_hits = {1: 0, 3: 0}
    comparisons = []

    for case in cases:
        question = case["question"]
        expected_source = case["expected_source"]
        expected_text = case["expected_text"]
        candidates = retriever.search(
            query=question,
            top_k=settings.retrieval_top_k,
        )
        reranked = reranker.rerank(
            query=question,
            chunks=candidates,
            top_k=len(candidates),
        )

        original_rank = find_rank(candidates, expected_source, expected_text)
        reranked_rank = find_rank(reranked, expected_source, expected_text)

        for k in before_hits:
            if original_rank is not None and original_rank <= k:
                before_hits[k] += 1
            if reranked_rank is not None and reranked_rank <= k:
                after_hits[k] += 1

        comparisons.append(
            {
                "question": question,
                "original_rank": original_rank,
                "reranked_rank": reranked_rank,
                "reranked": reranked,
            }
        )

    total = len(cases)
    print("Before Rerank:")
    print(f"Hit@1: {before_hits[1] / total:.3f} ({before_hits[1]}/{total})")
    print(f"Hit@3: {before_hits[3] / total:.3f} ({before_hits[3]}/{total})")

    print("\nAfter Rerank:")
    print(f"Hit@1: {after_hits[1] / total:.3f} ({after_hits[1]}/{total})")
    print(f"Hit@3: {after_hits[3] / total:.3f} ({after_hits[3]}/{total})")

    print("\nRank Changes:")
    for comparison in comparisons:
        print(
            f"{comparison['question']}: "
            f"{format_rank(comparison['original_rank'])} -> "
            f"{format_rank(comparison['reranked_rank'])}"
        )

        reranked_rank = comparison["reranked_rank"]
        if reranked_rank is not None and reranked_rank <= settings.rerank_top_k:
            continue

        print("  RERANK MISS")
        for final_rank, result in enumerate(comparison["reranked"], start=1):
            heading = result.chunk.text.split("\n\n", maxsplit=1)[0]
            preview = result.chunk.text.replace("\n", " ")[:160]
            print(
                "  ",
                f"final_rank={final_rank}",
                f"source={result.chunk.source}",
                f"chunk_id={result.chunk.chunk_id}",
                f"heading={heading}",
                f"retrieval_score={result.retrieval_score:.6f}",
                f"rerank_score={result.rerank_score:.6f}",
                f"preview={preview}",
            )


if __name__ == "__main__":
    main()
