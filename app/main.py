from fastapi import FastAPI

from app.api.dependencies import RAGServiceDep
from app.api.schemas import QueryRequest, QueryResponse
from app.core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest, service: RAGServiceDep):
    result = service.answer(question=request.question, top_k=request.top_k)

    return result
