from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, gt=0, le=10)


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
