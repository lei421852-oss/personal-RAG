from rag.retriever import RetrievedChunk

def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk]
) -> str:
    context_parts = []

    for index,result in enumerate(chunks,start=1):
        context_parts.append(
            f"[{index}]source:{result.chunk.source}\n"
            f"{result.chunk.text}"
        )

    context = "\n\n".join(context_parts)

    return (
        "Answer the question using only the provided context.\n"
        "If the context is insufficient, say that the knowledge base "
        "does not contain enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )