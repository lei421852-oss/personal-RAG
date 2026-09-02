import pytest

from app.core.config import Settings


def test_settings_use_non_sensitive_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.embedding_model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.raw_data_dir == "data/raw"
    assert settings.chroma_persist_directory == "data/chroma"
    assert settings.chroma_collection_name == "personal-rag"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100
    assert settings.retrieval_top_k == 10
    assert settings.rerank_top_k == 3
    assert settings.reranker_model_name == "BAAI/bge-reranker-base"
    assert settings.deepseek_model == "deepseek-v4-flash"

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is required"):
        settings.require_deepseek_api_key()


def test_settings_read_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")
    monkeypatch.setenv("RAG_RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "5")
    monkeypatch.setenv("RAG_CHROMA_COLLECTION_NAME", "test-collection")

    settings = Settings(_env_file=None)

    assert settings.require_deepseek_api_key() == "test-api-key"
    assert settings.retrieval_top_k == 8
    assert settings.rerank_top_k == 5
    assert settings.chroma_collection_name == "test-collection"
