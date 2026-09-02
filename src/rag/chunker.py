import re

from rag.document import Chunk, Document

HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


def _markdown_paragraphs(text: str) -> list[tuple[str | None, str]]:
    heading_levels: list[str | None] = [None, None, None]
    paragraphs: list[tuple[str | None, str]] = []
    current_lines: list[str] = []

    def heading_context() -> str | None:
        headings = [heading for heading in heading_levels if heading]
        return " > ".join(headings) if headings else None

    def flush_paragraph() -> None:
        paragraph = "\n".join(current_lines).strip()
        if paragraph:
            paragraphs.append((heading_context(), paragraph))
        current_lines.clear()

    for line in text.splitlines():
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_levels[level - 1] = heading_match.group(2).strip()
            for deeper_level in range(level, len(heading_levels)):
                heading_levels[deeper_level] = None
            continue

        if line.strip():
            current_lines.append(line.rstrip())
        else:
            flush_paragraph()

    flush_paragraph()
    return paragraphs


def _chunk_text(heading: str | None, body: str) -> str:
    if not heading:
        return body
    return f"{heading}\n\n{body}"


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    chunks = []

    paragraphs = _markdown_paragraphs(document.text)

    current_body = ""
    current_heading: str | None = None

    def append_chunk(body: str, heading: str | None) -> None:
        chunks.append(
            Chunk(
                text=_chunk_text(heading, body),
                source=document.source,
                chunk_id=f"{document.source}-{len(chunks)}",
            )
        )

    for heading, paragraph in paragraphs:
        if current_body and heading != current_heading:
            append_chunk(current_body, current_heading)
            current_body = ""

        current_heading = heading
        candidate_body = (
            paragraph if not current_body else current_body + "\n\n" + paragraph
        )
        if len(_chunk_text(heading, candidate_body)) <= chunk_size:
            current_body = candidate_body
            continue

        if current_body:
            append_chunk(current_body, current_heading)
            current_body = ""

        if len(_chunk_text(heading, paragraph)) <= chunk_size:
            current_body = paragraph
        else:
            heading_prefix = f"{heading}\n\n" if heading else ""
            body_size = chunk_size - len(heading_prefix)
            if body_size <= 0:
                max_heading_chars = max(0, chunk_size // 2 - 2)
                heading = heading[:max_heading_chars] if heading else None
                heading_prefix = f"{heading}\n\n" if heading else ""
                body_size = max(1, chunk_size - len(heading_prefix))

            effective_overlap = min(chunk_overlap, max(0, body_size - 1))
            step = body_size - effective_overlap

            for start in range(0, len(paragraph), step):
                body = paragraph[start : start + body_size]
                append_chunk(body, heading)

            current_body = ""

    if current_body:
        append_chunk(current_body, current_heading)

    return chunks
