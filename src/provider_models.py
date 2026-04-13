"""
Provider Models Module

This module handles provider and model configuration for the API-based paraphrase system.
It provides functions to resolve the appropriate model for OpenRouter through environment
files or defaults.
"""

from pathlib import Path
import os

DEFAULT_OPENROUTER_MODEL = "openrouter/free"

MODEL_FILE_OPENROUTER = Path.home() / ".model-openrouter"

def _read_single_line(path: Path, default_value: str) -> str:
    try:
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:
        pass
    return default_value

def model_for_provider(provider: str = "openrouter") -> str:
    return _read_single_line(MODEL_FILE_OPENROUTER, DEFAULT_OPENROUTER_MODEL)