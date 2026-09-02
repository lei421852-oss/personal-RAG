from rag.chunker import chunk_document
from rag.document import Document


def test_short_document_produces_one_chunk() -> None:
    document = Document(text="short document", source="short.md")

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "short document"
    assert chunks[0].source == "short.md"
    assert chunks[0].chunk_id == "short.md-0"


def test_paragraphs_that_fit_are_kept_in_one_chunk() -> None:
    document = Document(text="first paragraph\n\nsecond paragraph", source="doc.md")

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "first paragraph\n\nsecond paragraph"


def test_long_document_produces_multiple_chunks_with_metadata() -> None:
    document = Document(text="abcdefghijklmnopqrstuvwxyz", source="long.md")

    chunks = chunk_document(document, chunk_size=10, chunk_overlap=2)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert all(chunk.source == "long.md" for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [
        f"long.md-{index}" for index in range(len(chunks))
    ]
