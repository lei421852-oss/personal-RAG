from types import SimpleNamespace

import pytest
import torch

from rag.document import Chunk
from rag.reranker import BGEReranker
from rag.retriever import RetrievedChunk


class FakeTokenizer:
    def __init__(self):
        self.pairs: list[list[str]] | None = None

    def __call__(self, pairs: list[list[str]], **kwargs):
        self.pairs = pairs
        return {"input_ids": torch.tensor([[1], [2], [3]])}


class FakeSequenceModel:
    device = torch.device("cpu")

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[0.2], [1.5], [-0.1]]))


def test_bge_reranker_sorts_scores_and_preserves_retrieval_scores() -> None:
    candidates = [
        RetrievedChunk(
            chunk=Chunk(text=f"passage {index}", source="doc.md", chunk_id=str(index)),
            retrieval_score=score,
        )
        for index, score in enumerate([0.9, 0.8, 0.7])
    ]
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.tokenizer = FakeTokenizer()
    reranker.model = FakeSequenceModel()

    results = reranker.rerank(query="question", chunks=candidates, top_k=2)

    assert reranker.tokenizer.pairs == [
        ["question", "passage 0"],
        ["question", "passage 1"],
        ["question", "passage 2"],
    ]
    assert [result.chunk.chunk_id for result in results] == ["1", "0"]
    assert [result.retrieval_score for result in results] == [0.8, 0.9]
    assert [result.rerank_score for result in results] == pytest.approx([1.5, 0.2])
