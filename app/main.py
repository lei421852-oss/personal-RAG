import logging

from fastapi import FastAPI

from app.api.dependencies import RAGServiceDep, SettingsDep
from app.api.errors import (
    generation_error_handler,
    index_not_ready_error_handler,
    retrieval_error_handler,
)
from app.api.schemas import QueryRequest, QueryResponse
from app.core.lifespan import lifespan
from rag.errors import GenerationError, IndexNotReadyError, RetrievalError

logger = logging.getLogger(__name__)


def query(
    request: QueryRequest,
    service: RAGServiceDep,
    settings: SettingsDep,
):
    rerank_top_k = request.top_k or settings.rerank_top_k
    retrieval_top_k = max(settings.retrieval_top_k, rerank_top_k)
    logger.info(
        "query received question_chars=%d retrieval_top_k=%d rerank_top_k=%d",
        len(request.question),
        retrieval_top_k,
        rerank_top_k,
    )
    result = service.answer(
        question=request.question,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )

    return result


def create_app(*, lifespan_context=lifespan) -> FastAPI:
    application = FastAPI(lifespan=lifespan_context)
    application.add_exception_handler(GenerationError, generation_error_handler)
    application.add_exception_handler(RetrievalError, retrieval_error_handler)
    application.add_exception_handler(
        IndexNotReadyError,
        index_not_ready_error_handler,
    )
    application.post(
        "/query",
        response_model=QueryResponse,
    )(query)

    return application


app = create_app()
