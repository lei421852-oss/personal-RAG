from dataclasses import dataclass

from rag.llm import LLMCLient
from rag.prompt import build_rag_prompt
from rag.retriever import Retriever


@dataclass
class SourceInfo:
    source: str
    chunk_id: str
    score: float


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceInfo]


class RAGService:
    def __init__(self, retriver: Retriever, llm: LLMCLient):
        self.retriver = retriver
        self.llm = llm

    def answer(self, question: str, top_k: int = 3) -> RAGResult:
        results = self.retriver.search(query=question, top_k=top_k)

        prompt = build_rag_prompt(question=question, chunks=results)

        answer = self.llm.generate(prompt=prompt)

        sources = [
            SourceInfo(
                source=result.chunk.source,
                chunk_id=result.chunk.chunk_id,
                score=result.score,
            )
            for result in results
        ]

        return RAGResult(answer=answer, sources=sources)
