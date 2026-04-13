"""
Provider Facade Module

This module provides a unified interface for the OpenRouter API provider.
It routes generation and classification requests to the OpenRouter-specific implementations.
"""

from typing import Optional
from .provider_openrouter import generate_with_openrouter, classify_with_openrouter

def generate(text: str, prompt_template: Optional[str]) -> Optional[str]:
    from .provider_models import model_for_provider
    model = model_for_provider("openrouter")
    return generate_with_openrouter(text, model_name=model, prompt_template=prompt_template)

def classify(text: str, classification_prompt_template: Optional[str]) -> Optional[str]:
    from .provider_models import model_for_provider
    model = model_for_provider("openrouter")
    return classify_with_openrouter(text, model_name=model)