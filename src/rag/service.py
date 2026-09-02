import logging
from dataclasses import dataclass
from time import perf_counter

from rag.llm import LLMCLient
from rag.prompt import build_rag_prompt
from rag.reranker import Reranker
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class SourceInfo:
    source: str
    chunk_id: str
    retrieval_score: float
    rerank_score: float | None


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceInfo]


class RAGService:
    def __init__(self, retriver: Retriever, reranker: Reranker, llm: LLMCLient):
        self.retriver = retriver
        self.reranker = reranker
        self.llm = llm

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> RAGResult:
        total_started = perf_counter()
        retrieval_started = perf_counter()
        candidates = self.retriver.search(query=question, top_k=retrieval_top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        logger.info(
            "retrieval completed retrieved_count=%d retrieval_ms=%.2f",
            len(candidates),
            retrieval_ms,
        )

        rerank_started = perf_counter()
        results = self.reranker.rerank(
            query=question,
            chunks=candidates,
            top_k=rerank_top_k,
        )
        rerank_ms = (perf_counter() - rerank_started) * 1000
        logger.info(
            "reranking completed reranked_count=%d rerank_ms=%.2f",
            len(results),
            rerank_ms,
        )

        prompt = build_rag_prompt(question=question, chunks=results)

        generation_started = perf_counter()
        answer = self.llm.generate(prompt=prompt)
        generation_ms = (perf_counter() - generation_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "generation completed generation_ms=%.2f total_ms=%.2f",
            generation_ms,
            total_ms,
        )

        sources = [
            SourceInfo(
                source=result.chunk.source,
                chunk_id=result.chunk.chunk_id,
                retrieval_score=result.retrieval_score,
                rerank_score=result.rerank_score,
            )
            for result in results
        ]

        return RAGResult(answer=answer, sources=sources)
