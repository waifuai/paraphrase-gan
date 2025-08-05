import os
import pathlib
import logging
import pandas as pd
from typing import Optional
from google import genai
import time

def ensure_directory(directory_path):
    """Ensures that a directory exists. If not, it creates it."""
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
        logging.error(f"API key file not found at {expanded_path}.")
        raise
    except Exception as e:
        logging.error(f"Error loading API key from {expanded_path}: {e}")
        raise RuntimeError(f"Error loading API key: {e}")

# Initialize Gemini API client (call this once after loading key)
# api_key = load_gemini_api_key() # Load key first in main
# genai.configure(api_key=api_key)

def gemini_generate_paraphrase(input_text: str, client: genai.Client, model_name: str, prompt_template: str, max_retries: int = 3, delay: int = 5) -> Optional[str]:
    """
    Generates a paraphrase using Gemini API.

    Args:
        input_text: The original text to paraphrase.
        model: The configured Gemini model object (e.g., genai.GenerativeModel).
        prompt_template: A string template for the prompt, e.g., "Paraphrase this: {text}".
        max_retries: Maximum number of retries for API calls.
        delay: Initial delay between retries in seconds.

    Returns:
        The generated paraphrase text, or None if generation fails after retries.
    """
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
                return generated_text
            else:
                logging.warning(f"Attempt {attempt + 1} failed: No valid content in Gemini response for input '{input_text}'. Response: {response}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed during Gemini generation for input '{input_text}': {e}")

        if attempt < max_retries - 1:
            sleep_time = delay * (2 ** attempt)
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            logging.error(f"Failed to generate paraphrase after {max_retries} attempts for input '{input_text}'.")
            return None

    return None

def gemini_classify_paraphrase(text: str, client: genai.Client, model_name: str, prompt_template: str, max_retries: int = 3, delay: int = 5) -> Optional[str]:
    """
    Classifies text (e.g., paraphrase) using Gemini API.
    Assumes the prompt asks for a specific output like 'human' or 'machine'.

    Args:
        text: The text to classify.
        model: The configured Gemini model object (e.g., genai.GenerativeModel).
        prompt_template: A string template for the prompt, e.g., "Classify this: {text}".
        max_retries: Maximum number of retries for API calls.
        delay: Initial delay between retries in seconds.

    Returns:
        A normalized classification string ('human', 'machine', or 'error'), or None on final failure.
    """
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
                    return 'human'
                if 'machine' in classification_raw:
                    logging.debug("Classification successful: 'machine'")
                    return 'machine'
                logging.warning(f"Gemini response did not contain expected classification for text '{text}'. Response: '{classification_raw}'")
                return 'error'
            else:
                logging.warning(f"Attempt {attempt + 1} failed: No valid content in Gemini response for classification of '{text}'. Response: {response}")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed during Gemini classification for text '{text}': {e}")

        if attempt < max_retries - 1:
            sleep_time = delay * (2 ** attempt)
            logging.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            logging.error(f"Failed to classify text after {max_retries} attempts for '{text}'.")
            return None

    return None

def generate_mock_paraphrases(num_samples=100):
    """Generates mock data simulating input phrases."""
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

def postprocess_discriminator_output_gemini(generated_phrases_data):
    """
    Filters generated phrases based on Gemini classifications embedded in the data.
    Expects a list of dicts like {'input_text': ..., 'generated_text': ..., 'classification': ...}.
    Returns a list of dictionaries for selected pairs where classification was 'human'.
    """
    selected_pairs = []
    for item in generated_phrases_data:
        gen_text = item['generated_text'] # Fixed syntax error here
        original_text = item['input_text']
        classification = item.get('classification', 'error') # Default to error if missing

        # Normalize classification string to lower case and check
        if classification.lower() == 'human':
            selected_pairs.append({
                'input_text': original_text,
                'target_text': gen_text,
                'classification': classification # Keep for tracking
            })
            logging.debug(f"Selected phrase (classified as human): '{gen_text}' for input '{original_text}'")
        elif classification.lower() == 'error':
             logging.warning(f"Discarded phrase (classification error): '{gen_text}' for input '{original_text}'")
        else: # Assumes 'machine' or other
            logging.debug(f"Discarded phrase (classified as {classification}): '{gen_text}' for input '{original_text}'") # Added completion logic here

    logging.info(f"Post-processing complete. Selected {len(selected_pairs)} phrases classified as 'human'.") # Added completion logic here
    return selected_pairs # Added completion logic here