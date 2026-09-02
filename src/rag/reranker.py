from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag.retriever import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [[query, result.chunk.text] for result in chunks]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {
            name: tensor.to(self.model.device)
            for name, tensor in inputs.items()
        }

        with torch.no_grad():
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(-1)
                .float()
                .cpu()
                .tolist()
            )

        reranked = [
            RetrievedChunk(
                chunk=result.chunk,
                retrieval_score=result.retrieval_score,
                rerank_score=float(score),
            )
            for result, score in zip(chunks, scores, strict=True)
        ]
        reranked.sort(
            key=lambda result: result.rerank_score
            if result.rerank_score is not None
            else float("-inf"),
            reverse=True,
        )

        return reranked[:top_k]
