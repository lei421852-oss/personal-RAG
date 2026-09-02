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


def test_heading_is_included_with_paragraph_context() -> None:
    document = Document(
        text="## Lifespan 与资源生命周期\n\n模型在 startup 加载一次并复用。",
        source="lifespan.md",
    )

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == (
        "Lifespan 与资源生命周期\n\n模型在 startup 加载一次并复用。"
    )


def test_heading_switch_updates_following_chunk_context() -> None:
    document = Document(
        text=(
            "# Mini GPT Service\n\n"
            "## Lifespan\n\nstartup resources\n\n"
            "## Dependency Injection\n\ninjected service"
        ),
        source="architecture.md",
    )

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert [chunk.text for chunk in chunks] == [
        "Mini GPT Service > Lifespan\n\nstartup resources",
        "Mini GPT Service > Dependency Injection\n\ninjected service",
    ]


def test_markdown_without_heading_keeps_existing_behavior() -> None:
    document = Document(
        text="first paragraph\n\nsecond paragraph",
        source="plain.md",
    )

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=10)

    assert [chunk.text for chunk in chunks] == [
        "first paragraph\n\nsecond paragraph"
    ]
    assert chunks[0].source == "plain.md"
    assert chunks[0].chunk_id == "plain.md-0"


def test_long_paragraph_chunks_each_keep_heading() -> None:
    document = Document(
        text="## Lifespan\n\n" + "abcdefghijklmnopqrstuvwxyz",
        source="long-heading.md",
    )

    chunks = chunk_document(document, chunk_size=20, chunk_overlap=3)

    assert len(chunks) > 1
    assert all(chunk.text.startswith("Lifespan\n\n") for chunk in chunks)
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [
        f"long-heading.md-{index}" for index in range(len(chunks))
    ]
