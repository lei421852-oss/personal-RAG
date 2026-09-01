from typing import Annotated

from fastapi import Depends, Request

from rag.service import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


RAGServiceDep = Annotated[
    RAGService,
    Depends(get_rag_service),
]
