from typing import Protocol

from openai import OpenAI

class LLMCLient(Protocol):
    def generate(self,prompt: str) -> str:
        ...

class DeepSeekLLMClient:
    def __init__(
        self,
        api_key,
        model: str = "deepseek-v4-flash"
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        self.model = model

    def generate(self,prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        return response.output_text