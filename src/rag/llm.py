from typing import Protocol

from openai import OpenAI

from rag.errors import GenerationError


class LLMCLient(Protocol):
    def generate(self, prompt: str) -> str: ...


class DeepSeekLLMClient:
    def __init__(self, api_key, model: str = "deepseek-v4-flash"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.responses.create(model=self.model, input=prompt)
        except Exception as exc:
            raise GenerationError("Language model generation failed") from exc

        return response.output_text
