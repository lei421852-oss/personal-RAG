from typing import Protocol
import torch
import torch.nn.functional as F 
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel(Protocol):
    def _embed(self ,texts: list[str]) -> list[list[float]]:
        ...

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        ...

    def embed_query(
        self,
        query: str
    ) -> list[float]:
        ...

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    left_padding = (
        attention_mask[:,-1].sum()
        == attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden_states[:,-1]

    sequence_lengths = attention_mask.sum(dim=1) - 1

    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0]),
        sequence_lengths
    ]

class QwenEmbeddingModel:
    def __init__(
        self,
        model_name:str = "Qwen/Qwen3-Embedding-0.6B"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = last_token_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings.tolist()

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(
        self,
        query:str
    ) -> list[float]:
        instruction = (
            "Instruct: Given a ueser question,"
            "retrieve relevant passage that answer the question.\n"
            f"Query:{query}"
        )

        return self._embed([instruction])[0]
        