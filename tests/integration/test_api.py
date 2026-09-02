from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service
from app.core.config import Settings, get_settings
from app.main import create_app
from rag.errors import GenerationError, RetrievalError
from rag.service import RAGResult, SourceInfo


class FakeRAGService:
    def answer(
        self,
        question: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> RAGResult:
        return RAGResult(
            answer="fake answer",
            sources=[
                SourceInfo(
                    source="test.md",
                    chunk_id="test.md-0",
                    retrieval_score=0.82,
                    rerank_score=0.95,
                )
            ],
        )


class RaisingRAGService:
    def __init__(self, error: Exception):
        self.error = error

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> RAGResult:
        raise self.error


@asynccontextmanager
async def empty_lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


@pytest.fixture
def client() -> TestClient:
    application = create_app(lifespan_context=empty_lifespan)
    application.dependency_overrides[get_rag_service] = FakeRAGService
    application.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)

    with TestClient(application) as test_client:
        yield test_client


def test_query_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"question": "test question", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "fake answer",
        "sources": [
            {
                "source": "test.md",
                "chunk_id": "test.md-0",
                "retrieval_score": 0.82,
                "rerank_score": 0.95,
            }
        ],
    }


@pytest.mark.parametrize("top_k", [0, 100])
def test_query_rejects_invalid_top_k(client: TestClient, top_k: int) -> None:
    response = client.post(
        "/query",
        json={"question": "test question", "top_k": top_k},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (GenerationError("internal secret-api-key traceback"), 502),
        (RetrievalError("internal secret-api-key traceback"), 503),
    ],
)
def test_query_maps_domain_errors_without_leaking_details(
    error: Exception,
    expected_status: int,
) -> None:
    application = create_app(lifespan_context=empty_lifespan)
    application.dependency_overrides[get_rag_service] = lambda: RaisingRAGService(
        error
    )
    application.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)

    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/query",
            json={"question": "test question", "top_k": 3},
        )

    assert response.status_code == expected_status
    response_text = response.text.lower()
    assert "traceback" not in response_text
    assert "secret-api-key" not in response_text
    assert "internal" not in response_text
