from rag.document import Chunk, Document

def chunk_document(
        document: Document,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
) -> list[Chunk]:
    chunks = []

    paragraphs = [
        paragraph.strip()
        for paragraph in document.text.split("\n\n")
        if paragraph.strip()
    ]

    current_text = ""
    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current_text
            else current_text +"\n\n" + paragraph
        )
        if len(candidate) <= chunk_size:
            current_text = candidate
            continue

        if current_text:
            chunks.append(
                Chunk(
                    text=current_text,
                    source=document.source,
                    chunk_id=f"{document.source}-{len(chunks)}"
                )
            )

        if len(paragraph) <= chunk_size:
            current_text = paragraph
        else:
            step = chunk_size - chunk_overlap

            for start in range(0,len(paragraph),step):
                text = paragraph[start:start + chunk_size]
                chunks.append(
                    Chunk(
                        text=text,
                        source=document.source,
                        chunk_id=f"{document.source}-{len(chunks)}"
                    )
                )

            current_text = ""

    if current_text:
        chunks.append(
            Chunk(
                text=current_text,
                source=document.source,
                chunk_id=f"{document.source}-{len(chunks)}"
            )
        )
    return chunks