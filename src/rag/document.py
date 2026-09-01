from dataclasses import dataclass


@dataclass
class Document:
    text: str
    source: str


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str

