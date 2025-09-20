"""
Provider Models Module

This module handles provider and model configuration for the API-based paraphrase system.
It provides functions to determine which provider to use based on environment variables
and resolve the appropriate model for each provider.

The module supports both OpenRouter and Gemini providers, with configurable model
selection through environment files or defaults. It centralizes model management
to ensure consistent model selection across the application.
"""

from pathlib import Path
import os

DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

MODEL_FILE_OPENROUTER = Path.home() / ".model-openrouter"
MODEL_FILE_GEMINI = Path.home() / ".model-gemini"

def _read_single_line(path: Path, default_value: str) -> str:
    try:
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        pass
    return default_value

def provider_from_env() -> str:
    provider = os.getenv("PROVIDER", "").strip().lower()
    if provider in ("openrouter", "gemini"):
        return provider
    return "openrouter"

def model_for_provider(provider: str) -> str:
    if provider == "gemini":
        return _read_single_line(MODEL_FILE_GEMINI, DEFAULT_GEMINI_MODEL)
    return _read_single_line(MODEL_FILE_OPENROUTER, DEFAULT_OPENROUTER_MODEL)