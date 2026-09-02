from rag.retriever import RetrievedChunk


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []

    for index, result in enumerate(chunks, start=1):
        context_parts.append(
            f"[{index}] Source: {result.chunk.source}\n"
            f"Chunk: {result.chunk.chunk_id}\n"
            f"{result.chunk.text}"
        )

    context = "\n\n".join(context_parts)

    return (
        "You are a grounded question-answering assistant.\n"
        "Rules:\n"
        "1. Use only information explicitly supported by the provided context.\n"
        "2. Do not use your own knowledge to add or complete facts.\n"
        "3. If the context supports only part of the answer, answer only the "
        "supported part and clearly state what information is missing.\n"
        "4. If the context is completely insufficient, clearly state that the "
        "knowledge base does not contain enough information.\n"
        "5. Cite factual statements whenever possible with the supporting "
        "context number, such as [1] or [2].\n"
        "6. Do not cite a context number that does not exist below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
