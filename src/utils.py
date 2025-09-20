"""
Utilities Module

This module provides essential utility functions for the API-based paraphrase system including:
- Logging setup and configuration
- Directory creation utilities
- API key loading and management
- Gemini API integration functions for generation and classification
- Mock data generation for testing
- Post-processing of classification results
- Caching system for API responses to improve performance and reduce costs
"""

import os
import pathlib
import logging
import pandas as pd
import hashlib
import json
from typing import Optional, Dict, Any
from google import genai
import time

# Set up module-level logger
logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and hasn't expired."""
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return None

        return entry['value']

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache with TTL."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'value': value,
            'created_at': time.time(),
            'expires_at': time.time() + ttl_seconds
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


# Global cache instance - will be initialized with config in main
api_cache: Optional[SimpleCache] = None


def initialize_cache(config: Dict[str, Any]) -> None:
    """Initialize the global cache with configuration."""
    global api_cache
    cache_config = config.get('cache', {})
    if cache_config.get('enabled', True):
        api_cache = SimpleCache(max_size=cache_config.get('max_size', 1000))
        logging.info(f"API cache initialized with max size: {api_cache.max_size}")
    else:
        api_cache = None
        logging.info("API caching disabled")


def clear_api_cache() -> None:
    """Clear the API response cache."""
    api_cache.clear()
    logging.info("API cache cleared")


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        "cache_size": len(api_cache.cache),
        "max_cache_size": api_cache.max_size
    }

def setup_logger(logs_dir: pathlib.Path, config: Dict[str, Any]) -> logging.Logger:
    """
    Sets up the application logger with file and console handlers.

    Args:
        logs_dir: Directory where log files will be stored.
        config: Configuration dictionary containing logging settings.

    Returns:
        Configured logger instance.
    """
    # Get logger configuration
    log_config = config.get('logging', {})
    logger_name = log_config.get('logger_name', 'gemini_paraphrase')
    console_level = getattr(logging, log_config.get('console_level', 'INFO'))
    file_level = getattr(logging, log_config.get('file_level', 'DEBUG'))
    log_file = log_config.get('log_file', 'run.log')

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_path = logs_dir / log_file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

def ensure_directory(directory_path: pathlib.Path) -> None:
    """
    Ensures that a directory exists. If not, it creates it.

    Args:
        directory_path: Path object representing the directory to create.

    Returns:
        None
    """
    os.makedirs(directory_path, exist_ok=True)

def load_gemini_api_key(api_key_path="~/.api-gemini") -> str:
    """
    Loads Gemini API key preferring env vars, with file fallback.
    Env vars: GEMINI_API_KEY or GOOGLE_API_KEY.
    """
    # Env-first
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    # Fallback file
    expanded_path = pathlib.Path(api_key_path).expanduser()
    try:
        with open(expanded_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty.")
            logging.info(f"Loaded API key from {expanded_path}")
            return api_key
    except FileNotFoundError:
        logging.error(f"API key file not found at {expanded_path}. Please ensure the file exists and the path is correct.")
        raise FileNotFoundError(f"API key file not found at {expanded_path}")
    except PermissionError:
        logging.error(f"Permission denied when trying to read API key file at {expanded_path}. Please check file permissions.")
        raise PermissionError(f"Permission denied for API key file at {expanded_path}")
    except Exception as e:
        logging.error(f"Unexpected error loading API key from {expanded_path}: {e}")
        logging.debug(f"Exception type: {type(e).__name__}", exc_info=True)
        raise RuntimeError(f"Error loading API key: {e}")

# Initialize Gemini API client (call this once after loading key)
# api_key = load_gemini_api_key() # Load key first in main
# genai.configure(api_key=api_key)

def gemini_generate_paraphrase(input_text: str, client: genai.Client, model_name: str, prompt_template: str, max_retries: int = 3, delay: int = 5, use_cache: bool = True) -> Optional[str]:
    """
    Generates a paraphrase using Gemini API.

    Args:
        input_text: The original text to paraphrase.
        model: The configured Gemini model object (e.g., genai.GenerativeModel).
        prompt_template: A string template for the prompt, e.g., "Paraphrase this: {text}".
        max_retries: Maximum number of retries for API calls.
        delay: Initial delay between retries in seconds.
        use_cache: Whether to use caching for API responses.

    Returns:
        The generated paraphrase text, or None if generation fails after retries.
    """
    # Check cache first
    cache_key = None
    if use_cache and api_cache:
        cache_key = api_cache._generate_key('generate', input_text, model_name, prompt_template)
        cached_result = api_cache.get(cache_key)
        if cached_result:
            logging.debug(f"Cache hit for input: '{input_text}'")
            return cached_result

    prompt = prompt_template.format(text=input_text)
    logging.debug(f"Generating paraphrase for '{input_text}' with prompt: '{prompt}'")

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # Prefer robust parsing of candidates
            text: Optional[str] = None
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    part = cand.content.parts[0]
                    text = getattr(part, "text", None)
            # Some SDKs expose convenience .text
            if not text and hasattr(response, "text"):
                text = getattr(response, "text", None)

            if text:
                generated_text = text.strip()
                logging.debug(f"Generation successful: '{generated_text}'")

                # Cache the result
                if use_cache and api_cache and cache_key:
                    api_cache.set(cache_key, generated_text)

                return generated_text
            else:
                logging.warning(f"Attempt {attempt + 1} failed: No valid content in Gemini response for input '{input_text}'. Response: {response}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed during Gemini generation for input '{input_text}': {e}")
            logging.debug(f"Exception type: {type(e).__name__}, Exception details: {str(e)}", exc_info=True)

        if attempt < max_retries - 1:
            sleep_time = delay * (2 ** attempt)
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            logging.error(f"Failed to generate paraphrase after {max_retries} attempts for input '{input_text}'. All retries exhausted.")
            return None

    return None

def gemini_classify_paraphrase(text: str, client: genai.Client, model_name: str, prompt_template: str, max_retries: int = 3, delay: int = 5, use_cache: bool = True) -> Optional[str]:
    """
    Classifies text (e.g., paraphrase) using Gemini API.
    Assumes the prompt asks for a specific output like 'human' or 'machine'.

    Args:
        text: The text to classify.
        model: The configured Gemini model object (e.g., genai.GenerativeModel).
        prompt_template: A string template for the prompt, e.g., "Classify this: {text}".
        max_retries: Maximum number of retries for API calls.
        delay: Initial delay between retries in seconds.
        use_cache: Whether to use caching for API responses.

    Returns:
        A normalized classification string ('human', 'machine', or 'error'), or None on final failure.
    """
    # Check cache first
    cache_key = None
    if use_cache and api_cache:
        cache_key = api_cache._generate_key('classify', text, model_name, prompt_template)
        cached_result = api_cache.get(cache_key)
        if cached_result:
            logging.debug(f"Cache hit for classification: '{text[:50]}...'")
            return cached_result

    prompt = prompt_template.format(text=text)
    logging.debug(f"Classifying text '{text}' with prompt: '{prompt}'")

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # Parse text similarly to generation
            parsed_text: Optional[str] = None
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    part = cand.content.parts[0]
                    parsed_text = getattr(part, "text", None)
            if not parsed_text and hasattr(response, "text"):
                parsed_text = getattr(response, "text", None)

            if parsed_text:
                classification_raw = parsed_text.strip().lower()
                if 'human' in classification_raw:
                    logging.debug("Classification successful: 'human'")
                    if use_cache and api_cache and cache_key:
                        api_cache.set(cache_key, 'human')
                    return 'human'
                if 'machine' in classification_raw:
                    logging.debug("Classification successful: 'machine'")
                    if use_cache and api_cache and cache_key:
                        api_cache.set(cache_key, 'machine')
                    return 'machine'
                logging.warning(f"Gemini response did not contain expected classification for text '{text}'. Response: '{classification_raw}'")
                if use_cache and api_cache and cache_key:
                    api_cache.set(cache_key, 'error')
                return 'error'
            else:
                logging.warning(f"Attempt {attempt + 1} failed: No valid content in Gemini response for classification of '{text}'. Response: {response}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed during Gemini classification for text '{text}': {e}")
            logging.debug(f"Exception type: {type(e).__name__}, Exception details: {str(e)}", exc_info=True)

        if attempt < max_retries - 1:
            sleep_time = delay * (2 ** attempt)
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            logging.error(f"Failed to classify text after {max_retries} attempts for '{text}'. All retries exhausted.")
            return None

    return None

def generate_mock_paraphrases(num_samples: int = 100) -> pd.DataFrame:
    """
    Generates mock data simulating input phrases for testing.

    Args:
        num_samples: Number of mock input phrases to generate.

    Returns:
        A pandas DataFrame with 'input_text' column containing the mock phrases.
    """
    mock_gen_data = []
    base_phrases = [
        "hello world", "how are you", "nice to meet you", "goodbye", "thank you",
        "please help me", "what is your name", "tell me a story", "have a nice day", "see you later"
    ]
    for i in range(num_samples):
        original = base_phrases[i % len(base_phrases)]
        # Only need input data for the API approach
        mock_gen_data.append({'input_text': original})

    gen_df = pd.DataFrame(mock_gen_data).drop_duplicates(subset=['input_text']).reset_index(drop=True)
    return gen_df

def postprocess_discriminator_output_gemini(generated_phrases_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filters generated phrases based on Gemini classifications embedded in the data.

    Args:
        generated_phrases_data: List of dictionaries containing 'input_text', 'generated_text', and 'classification'.

    Returns:
        A list of dictionaries for selected pairs where classification was 'human',
        with keys 'input_text', 'target_text', and 'classification'.
    """
    selected_pairs = []
    for item in generated_phrases_data:
        gen_text = item['generated_text']
        original_text = item['input_text']
        classification = item.get('classification', 'error')  # Default to error if missing

        # Normalize classification string to lower case and check
        if classification.lower() == 'human':
            selected_pairs.append({
                'input_text': original_text,
                'target_text': gen_text,
                'classification': classification  # Keep for tracking
            })
            logging.debug(f"Selected phrase (classified as human): '{gen_text}' for input '{original_text}'")
        elif classification.lower() == 'error':
            logging.warning(f"Discarded phrase (classification error): '{gen_text}' for input '{original_text}'")
        else:  # Assumes 'machine' or other
            logging.debug(f"Discarded phrase (classified as {classification}): '{gen_text}' for input '{original_text}'")

    logging.info(f"Post-processing complete. Selected {len(selected_pairs)} phrases classified as 'human'.")
    return selected_pairs