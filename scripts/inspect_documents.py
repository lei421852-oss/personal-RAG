from rag.loader import load_markdown_documents
from rag.chunker import chunk_document

documents = load_markdown_documents("data/raw")

for document in documents:
    chunks = chunk_document(document, chunk_size=500)
    print(document.source)
    print("num_chunk:", len(chunks))
    for chunk in chunks[:3]:
        print(chunk.chunk_id)
        print(chunk.text[:100])
        print("=" * 30)
    print(len(document.text))
    print(document.text[:100])
    print("-" * 100)
