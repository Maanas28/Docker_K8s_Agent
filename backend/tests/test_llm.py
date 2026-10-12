"""OllamaLLM error handling, with the Ollama client replaced by a fake."""

import httpx
import pytest
from ollama import ResponseError

from app.llm import LLMUnavailable, OllamaLLM


class FailingClient:
    def __init__(self, error: Exception, when: str = "call"):
        self.error, self.when = error, when

    async def chat(self, **kwargs):
        if self.when == "call":
            raise self.error
        return self._stream()

    async def _stream(self):
        raise self.error
        yield  # makes this an async generator


def make(error, when="call"):
    llm = OllamaLLM("http://localhost:11434", "qwen2.5-coder:7b")
    llm._client = FailingClient(error, when)
    return llm


async def drain(llm):
    return [p async for p in llm.stream([], None)]


@pytest.mark.parametrize("when", ["call", "iterate"])
async def test_missing_model_says_how_to_fix_it(when):
    llm = make(ResponseError("model 'qwen2.5-coder:7b' not found", 404), when)
    with pytest.raises(LLMUnavailable, match=r"isn't downloaded\. Run `ollama pull qwen2\.5-coder:7b`"):
        await drain(llm)
    if when == "call":
        with pytest.raises(LLMUnavailable, match="isn't downloaded"):
            await llm.classify([])


async def test_ollama_down_says_how_to_start_it():
    llm = make(httpx.ConnectError("refused"))
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        await llm.classify([])


async def test_other_ollama_errors_are_not_hidden():
    llm = make(ResponseError("bad request", 400))
    with pytest.raises(ResponseError):
        await llm.classify([])
