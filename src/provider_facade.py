"""
Provider Facade Module

This module provides a unified interface for different API providers (OpenRouter and Gemini).
It acts as a facade that abstracts away the differences between providers, allowing the
main application to use a consistent API regardless of which provider is selected.

The facade handles provider selection based on environment variables and routes generation
and classification requests to the appropriate provider-specific implementations.
"""

from typing import Optional
from google import genai
from .provider_models import provider_from_env, model_for_provider
from .provider_openrouter import generate_with_openrouter, classify_with_openrouter
from .utils import gemini_generate_paraphrase, gemini_classify_paraphrase

def generate(text: str, prompt_template: Optional[str], client: Optional[genai.Client]) -> Optional[str]:
    provider = provider_from_env()
    model = model_for_provider(provider)
    if provider == "openrouter":
        return generate_with_openrouter(text, model_name=model, prompt_template=prompt_template)
    if provider == "gemini":
        if client is None:
            return None
        return gemini_generate_paraphrase(
            input_text=text,
            client=client,
            model_name=model,
            prompt_template=prompt_template or "Paraphrase: {text}",
            max_retries=3,
            delay=5,
        )
    return None

def classify(text: str, classification_prompt_template: Optional[str], client: Optional[genai.Client]) -> Optional[str]:
    provider = provider_from_env()
    model = model_for_provider(provider)
    if provider == "openrouter":
        return classify_with_openrouter(text, model_name=model)
    if provider == "gemini":
        if client is None:
            return None
        return gemini_classify_paraphrase(
            text=text,
            client=client,
            model_name=model,
            prompt_template=classification_prompt_template or "Classify: {text}",
            max_retries=3,
            delay=5,
        )
    return None