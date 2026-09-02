from rag.document import Chunk
from rag.prompt import build_rag_prompt
from rag.retriever import RetrievedChunk


def test_prompt_contains_question_context_sources_and_guardrail() -> None:
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                text="Lifespan loads shared resources during application startup.",
                source="architecture.md",
                chunk_id="architecture.md-0",
            ),
            retrieval_score=0.9,
        ),
        RetrievedChunk(
            chunk=Chunk(
                text="Requests reuse the initialized resources.",
                source="runtime.md",
                chunk_id="runtime.md-1",
            ),
            retrieval_score=0.8,
        ),
    ]

    prompt = build_rag_prompt(
        question="Why use FastAPI lifespan?",
        chunks=chunks,
    )

    assert "Why use FastAPI lifespan?" in prompt
    assert "Lifespan loads shared resources" in prompt
    assert "Requests reuse the initialized resources" in prompt
    assert "architecture.md" in prompt
    assert "runtime.md" in prompt
    assert "[1] Source: architecture.md" in prompt
    assert "Chunk: architecture.md-0" in prompt
    assert "only information explicitly supported" in prompt
    assert "Do not use your own knowledge" in prompt
    assert "supports only part of the answer" in prompt
    assert "knowledge base does not contain enough information" in prompt
    assert "such as [1] or [2]" in prompt
    assert "Do not cite a context number that does not exist" in prompt
