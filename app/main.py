from fastapi import FastAPI

from app.api.dependencies import RAGServiceDep, SettingsDep
from app.api.schemas import QueryRequest, QueryResponse
from app.core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    service: RAGServiceDep,
    settings: SettingsDep,
):
    top_k = request.top_k or settings.default_top_k
    result = service.answer(question=request.question, top_k=top_k)

    return result
