class RAGError(Exception):
    """Base exception for expected RAG failures."""


class IndexNotReadyError(RAGError):
    """Raised when the persistent vector index is missing or empty."""


class RetrievalError(RAGError):
    """Raised when the retrieval infrastructure cannot complete a query."""


class GenerationError(RAGError):
    """Raised when the external LLM provider cannot generate an answer."""
