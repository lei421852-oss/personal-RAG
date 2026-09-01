from pathlib import Path

from rag.document import Document


def load_markdown_documents(data_dir) -> list[Document]:
    documents = []

    for path in Path(data_dir).glob("*.md"):
        text = path.read_text(encoding="utf-8")

        documents.append(
            Document(
                text=text,
                source=path.name,
            )
        )

    return documents
