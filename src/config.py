import logging
from pathlib import Path
from typing import Dict, Any

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define project root based on this file's location
PROJECT_ROOT = Path(__file__).parent.parent


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_str(key: str, default: str) -> str:
    """Get string value from environment variable."""
    return os.getenv(key, default)

CONFIG: Dict[str, Any] = {
    "paths": {
        "project_root": PROJECT_ROOT,
        "config_dir": PROJECT_ROOT / "config", # Keep for potential future config files
        "scripts_dir": PROJECT_ROOT / "scripts",
        "src_dir": PROJECT_ROOT / "src",
        "data_root": PROJECT_ROOT / "data",
        "logs_dir": PROJECT_ROOT / "logs",
        "raw_data_dir": PROJECT_ROOT / "data" / "raw", # Where mock input might be saved/loaded
        "processed_data_dir": PROJECT_ROOT / "data" / "processed", # Where results are saved
        "selected_paraphrases_dir": PROJECT_ROOT / "data" / "processed" / "selected", # Store selected paraphrases
        "prompt_history_dir": PROJECT_ROOT / "data" / "processed" / "prompts", # Store prompt evolution
        "api_key_file": Path.home() / ".api-gemini", # Standard location for API key
    },
    "filenames": {
        "log_file": "run_gemini.log", # New log file name
        "mock_input_data": "mock_input_phrases.tsv", # Simple input file
        "selected_paraphrases": "selected_paraphrases_{iteration}.tsv", # Iteration-based output
        "prompt_history": "generator_prompt_{iteration}.txt", # Track prompts
        "loop_results": "loop_results_{iteration}.json", # Store metrics/results per iteration
    },
    "gemini": {
        "model_name": get_env_str('GEMINI_MODEL', "gemini-2.5-pro"),  # Or the specific model you want to use
        "generation_prompt_template": get_env_str(
            'GENERATION_PROMPT_TEMPLATE',
            "Generate a concise paraphrase for the following phrase, focusing on maintaining the original meaning but using different wording:\n\nPhrase: {text}\nParaphrase:"
        ),
        "classification_prompt_template": get_env_str(
            'CLASSIFICATION_PROMPT_TEMPLATE',
            "Does the following phrase sound like natural, fluent, human-written English? Answer only 'human' or 'machine'.\n\nPhrase: {text}\nClassification:"
        ),
        "max_retries": get_env_int('API_MAX_RETRIES', 3),
        "retry_delay": get_env_int('API_RETRY_DELAY', 5),  # Initial delay in seconds
    },
    "loop_control": {
        "max_iterations": get_env_int('MAX_ITERATIONS', 10),  # Limit the number of iterations for testing/cost control
        "mock_data_samples": get_env_int('MOCK_DATA_SAMPLES', 50),  # Number of input phrases per iteration
        "batch_size": get_env_int('BATCH_SIZE', 5),  # How many phrases to process before potential sleep/rate limit check
        "sleep_between_batches": get_env_int('SLEEP_BETWEEN_BATCHES', 1),  # Seconds to sleep between batches (adjust based on API limits)
        "prompt_refinement_strategy": get_env_str('PROMPT_REFINEMENT_STRATEGY', "basic_feedback"),  # Placeholder for future refinement logic
    },
    "logging": {
        "logger_name": get_env_str('LOGGER_NAME', "gemini_paraphrase"),
        "console_level": getattr(logging, get_env_str('CONSOLE_LOG_LEVEL', 'INFO')),
        "file_level": getattr(logging, get_env_str('FILE_LOG_LEVEL', 'DEBUG')),
    },
    "cache": {
        "enabled": get_env_bool('API_CACHE_ENABLED', True),
        "max_size": get_env_int('API_CACHE_MAX_SIZE', 1000),
        "ttl_seconds": get_env_int('API_CACHE_TTL', 300),  # 5 minutes default
    }
}

# --- Prompt Refinement Logic (Placeholder) ---
# This section could be expanded significantly based on the chosen strategy

def refine_generator_prompt(current_prompt: str, results: Dict[str, Any], strategy: str) -> str:
    """
    Refines the generator prompt based on the results of the last iteration.
    This is a placeholder and needs actual implementation.

    Args:
        current_prompt: The current generator prompt string.
        results: Dictionary containing results from the last iteration.
        strategy: The refinement strategy to use.

    Returns:
        The refined generator prompt string.
    """
    logger.info(f"Refining generator prompt using strategy: {strategy}")
    logger.debug(f"Current prompt: {current_prompt}")
    logger.debug(f"Results from last iteration: {results}")

    new_prompt = current_prompt # Default: no change

    if strategy == "basic_feedback":
        # Example: If too many 'machine' classifications, add emphasis on sounding natural
        selection_rate = results.get('selection_rate', 1.0) # Default to 1.0 if not found
        if selection_rate < 0.5: # Arbitrary threshold
            logger.info("Low selection rate detected. Adding emphasis on naturalness to prompt.")
            if "sound natural" not in new_prompt.lower():
                # Find the instruction part and append
                parts = new_prompt.split('\n\nPhrase:')
                if len(parts) > 1:
                    parts[0] += " Ensure the paraphrase sounds natural and fluent."
                    new_prompt = '\n\nPhrase:'.join(parts)
                else: # Fallback if structure is unexpected
                    new_prompt += "\n\nInstruction: Ensure the paraphrase sounds natural and fluent."

    # Add more sophisticated strategies here based on analysis of rejected/accepted phrases

    if new_prompt != current_prompt:
        logger.info(f"Refined prompt: {new_prompt}")
    else:
        logger.info("No changes made to the prompt for this iteration.")

    return new_prompt