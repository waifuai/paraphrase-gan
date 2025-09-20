from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_KEY_FILE_PATH = Path.home() / ".api-openrouter"

def _resolve_openrouter_api_key() -> Optional[str]:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        if OPENROUTER_API_KEY_FILE_PATH.is_file():
            return OPENROUTER_API_KEY_FILE_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

def _generation_prompt(text: str, template: Optional[str]) -> str:
    if template and "{text}" in template:
        return template.format(text=text)
    return (
        "Generate a concise paraphrase for the following phrase, preserving meaning and fluency.\n\n"
        f"Phrase: \"{text}\"\n\nParaphrase:"
    )

def _classification_prompt(text: str) -> str:
    return (
        "Classify the following sentence as human-written or machine-generated.\n\n"
        "Categories:\n"
        "- '1': Human-written\n"
        "- '0': Machine-generated\n\n"
        "Respond with ONLY the digit '0' or '1'.\n\n"
        f"Sentence: \"{text}\"\n\n"
        "Classification:"
    )

def _post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

def generate_with_openrouter(text: str, model_name: str = DEFAULT_OPENROUTER_MODEL, prompt_template: Optional[str] = None, timeout: int = 60) -> Optional[str]:
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        return None
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": _generation_prompt(text, prompt_template)}],
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post(OPENROUTER_API_URL, headers, payload, timeout)
    if not data:
        return None
    choices = data.get("choices", [])
    if not choices:
        return None
    content = (choices[0].get("message", {}).get("content") or "").strip()
    return content or None

def classify_with_openrouter(text: str, model_name: str = DEFAULT_OPENROUTER_MODEL, timeout: int = 60) -> Optional[str]:
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        return None
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": _classification_prompt(text)}],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post(OPENROUTER_API_URL, headers, payload, timeout)
    if not data:
        return None
    choices = data.get("choices", [])
    if not choices:
        return None
    content = (choices[0].get("message", {}).get("content") or "").strip()
    if not content:
        return None
    if content in ("0", "1"):
        return content
    contains0 = "0" in content
    contains1 = "1" in content
    if contains0 and not contains1:
        return "0"
    if contains1 and not contains0:
        return "1"
    return None