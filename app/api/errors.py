import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from rag.errors import GenerationError, IndexNotReadyError, RetrievalError

logger = logging.getLogger(__name__)


async def generation_error_handler(
    request: Request,
    exc: GenerationError,
) -> JSONResponse:
    logger.error("LLM generation failed error_type=%s", type(exc).__name__)
    return JSONResponse(
        status_code=502,
        content={"detail": "The language model service is temporarily unavailable."},
    )


async def retrieval_error_handler(
    request: Request,
    exc: RetrievalError,
) -> JSONResponse:
    logger.error("Retrieval failed error_type=%s", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={"detail": "RAG service is temporarily unavailable."},
    )


async def index_not_ready_error_handler(
    request: Request,
    exc: IndexNotReadyError,
) -> JSONResponse:
    logger.error("RAG index is not ready error_type=%s", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={"detail": "RAG service is temporarily unavailable."},
    )
