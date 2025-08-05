import pytest
from types import SimpleNamespace

from src.utils import (
    gemini_generate_paraphrase,
    gemini_classify_paraphrase,
)

class DummyClient:
    class ModelsNS:
        def __init__(self, response_text: str):
            self._text = response_text
        def generate_content(self, model: str, contents: str):
            # mimic google-genai response shape with candidates[0].content.parts[0].text and .text
            part = SimpleNamespace(text=self._text)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate], text=self._text)
    def __init__(self, response_text: str):
        self.models = DummyClient.ModelsNS(response_text)

def test_utils_generation_and_classification_mocked():
    client = DummyClient("human")  # will return "human" for both calls
    model_name = "gemini-2.5-pro"
    gen = gemini_generate_paraphrase(
        input_text="hello world",
        client=client,
        model_name=model_name,
        prompt_template="Paraphrase: {text}",
        max_retries=1,
        delay=0,
    )
    assert isinstance(gen, str) and len(gen) > 0

    cls = gemini_classify_paraphrase(
        text="anything",
        client=client,
        model_name=model_name,
        prompt_template="Classify: {text}",
        max_retries=1,
        delay=0,
    )
    assert cls == "human"