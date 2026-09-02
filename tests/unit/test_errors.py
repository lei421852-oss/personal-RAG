import pytest

from rag.errors import GenerationError, RetrievalError
from rag.llm import DeepSeekLLMClient
from rag.vector_store import ChromaVectorStore


class FailingResponses:
    def create(self, *, model: str, input: str) -> None:
        raise RuntimeError("provider failure")


class FailingLLMSDK:
    responses = FailingResponses()


class FailingCollection:
    def count(self) -> int:
        return 1

    def query(self, **kwargs) -> None:
        raise RuntimeError("vector database failure")


def test_llm_provider_failure_becomes_generation_error() -> None:
    llm = DeepSeekLLMClient.__new__(DeepSeekLLMClient)
    llm.client = FailingLLMSDK()
    llm.model = "test-model"

    with pytest.raises(GenerationError) as captured:
        llm.generate("sensitive prompt")

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "sensitive prompt" not in str(captured.value)


def test_vector_query_failure_becomes_retrieval_error() -> None:
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = FailingCollection()

    with pytest.raises(RetrievalError) as captured:
        store.search(query_embedding=[0.1, 0.2], top_k=1)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "vector database failure" not in str(captured.value)
