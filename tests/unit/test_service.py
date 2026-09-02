from rag.document import Chunk
from rag.retriever import RetrievedChunk
from rag.service import RAGService


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]):
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


class FakeReranker:
    def __init__(self):
        self.last_query: str | None = None
        self.last_chunks: list[RetrievedChunk] | None = None
        self.last_top_k: int | None = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.last_query = query
        self.last_chunks = chunks
        self.last_top_k = top_k
        return [
            RetrievedChunk(
                chunk=result.chunk,
                retrieval_score=result.retrieval_score,
                rerank_score=1.25,
            )
            for result in chunks[:top_k]
        ]


def test_answer_uses_retrieval_context_and_returns_sources() -> None:
    retrieved = RetrievedChunk(
        chunk=Chunk(
            text="Use lifespan to initialize shared resources once.",
            source="architecture.md",
            chunk_id="architecture.md-2",
        ),
        retrieval_score=0.87,
    )
    retriever = FakeRetriever([retrieved])
    reranker = FakeReranker()
    llm = FakeLLMClient(response="fake answer")
    service = RAGService(retriver=retriever, reranker=reranker, llm=llm)

    result = service.answer(
        question="Why lifespan?",
        retrieval_top_k=10,
        rerank_top_k=3,
    )

    assert retriever.last_query == "Why lifespan?"
    assert retriever.last_top_k == 10
    assert reranker.last_query == "Why lifespan?"
    assert reranker.last_chunks == [retrieved]
    assert reranker.last_top_k == 3
    assert llm.last_prompt is not None
    assert "Why lifespan?" in llm.last_prompt
    assert retrieved.chunk.text in llm.last_prompt
    assert retrieved.chunk.source in llm.last_prompt
    assert result.answer == "fake answer"
    assert len(result.sources) == 1
    assert result.sources[0].source == "architecture.md"
    assert result.sources[0].chunk_id == "architecture.md-2"
    assert result.sources[0].retrieval_score == 0.87
    assert result.sources[0].rerank_score == 1.25
