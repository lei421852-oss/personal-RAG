from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        validation_alias="RAG_EMBEDDING_MODEL_NAME",
    )
    raw_data_dir: str = Field(
        default="data/raw",
        validation_alias="RAG_RAW_DATA_DIR",
    )
    chroma_persist_directory: str = Field(
        default="data/chroma",
        validation_alias="RAG_CHROMA_PERSIST_DIRECTORY",
    )
    chroma_collection_name: str = Field(
        default="personal-rag",
        validation_alias="RAG_CHROMA_COLLECTION_NAME",
    )
    chunk_size: int = Field(
        default=500,
        gt=0,
        validation_alias="RAG_CHUNK_SIZE",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        validation_alias="RAG_CHUNK_OVERLAP",
    )
    retrieval_top_k: int = Field(
        default=10,
        gt=0,
        validation_alias="RAG_RETRIEVAL_TOP_K",
    )
    rerank_top_k: int = Field(
        default=3,
        gt=0,
        le=10,
        validation_alias="RAG_RERANK_TOP_K",
    )
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias="RAG_RERANKER_MODEL_NAME",
    )
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias="DEEPSEEK_MODEL",
    )

    def require_deepseek_api_key(self) -> str:
        if not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required to start the FastAPI application"
            )

        return self.deepseek_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
