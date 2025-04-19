import sys
import time
import logging
from typing import Dict, Any
import pandas as pd
import google.generativeai as genai

# Import components from the new modules
from .config import CONFIG
from .utils import (
    logger, # Logger is now initialized in utils
    ensure_directory,
    load_gemini_api_key,
    generate_mock_paraphrases # Mock data generation using pandas
)
# Import the loop iteration function from the renamed file
from .prompt_loop import run_prompt_refinement_iteration

# --- Main Script Execution ---
def main(config: Dict[str, Any]):
    """Main function to execute the Gemini prompt refinement workflow."""
    logger.info("========== Starting Gemini Paraphrase Prompt Refinement ==========")

    # --- Initial Setup ---
    paths = config['paths']
    filenames = config['filenames']
    gemini_config = config['gemini']
    loop_config = config['loop_control']

    # Create essential directory structure from config
    ensure_directory(paths['logs_dir'])
    ensure_directory(paths['raw_data_dir'])
    ensure_directory(paths['processed_data_dir'])
    ensure_directory(paths['selected_paraphrases_dir'])
    ensure_directory(paths['prompt_history_dir'])

    # --- Load API Key and Configure Gemini ---
    try:
        api_key = load_gemini_api_key(str(paths['api_key_file'])) # Pass path from config
        genai.configure(api_key=api_key)
        logger.info(f"Configuring Gemini model: {gemini_config['model_name']}")
        # Configure the specific model (add generation/safety settings if needed)
        gemini_model = genai.GenerativeModel(gemini_config['model_name'])
        logger.info("Gemini API configured successfully.")
    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        logger.critical(f"Failed to initialize Gemini API: {e}. Please ensure your API key is correctly placed and valid.")
        sys.exit(1)

    # --- Load or Generate Input Data ---
    input_data_path = paths['raw_data_dir'] / filenames['mock_input_data']
    input_df = None
    if input_data_path.exists():
        try:
            input_df = pd.read_csv(input_data_path, sep='\t')
            # Validate expected column
            if 'input_text' not in input_df.columns:
                 logger.error(f"Input file {input_data_path} missing required 'input_text' column.")
                 raise ValueError("Missing 'input_text' column")
            logger.info(f"Loaded {len(input_df)} input phrases from {input_data_path}")
            # Optionally limit samples if file is large
            if len(input_df) > loop_config['mock_data_samples']:
                logger.warning(f"Input file has {len(input_df)} samples, using first {loop_config['mock_data_samples']} as configured.")
                input_df = input_df.head(loop_config['mock_data_samples'])

        except Exception as e:
            logger.error(f"Failed to load or validate input data from {input_data_path}: {e}. Will attempt to generate mock data.")
            input_df = None # Reset to trigger mock generation

    if input_df is None:
        logger.info(f"Generating {loop_config['mock_data_samples']} mock input phrases...")
        try:
            input_df = generate_mock_paraphrases(loop_config['mock_data_samples'])
            # Save the generated mock data for future runs
            ensure_directory(paths['raw_data_dir'])
            input_df.to_csv(input_data_path, sep='\t', index=False)
            logger.info(f"Saved generated mock data to {input_data_path}")
        except Exception as e:
            logger.critical(f"Failed to generate mock data: {e}")
            sys.exit(1)

    # --- Prompt Refinement Loop ---
    current_prompt = gemini_config['generation_prompt_template']
    max_iterations = loop_config['max_iterations']

    for iteration in range(1, max_iterations + 1):
        logger.info(f"========== Starting Prompt Loop Iteration {iteration}/{max_iterations} ==========")
        try:
            iteration_results, next_prompt = run_prompt_refinement_iteration(
                iteration=iteration,
                current_generator_prompt=current_prompt,
                gemini_model=gemini_model,
                input_data=input_df, # Pass the DataFrame
                config=config
            )
            current_prompt = next_prompt # Update prompt for the next loop
            logger.info(f"Iteration {iteration} complete. Prompt for next iteration updated.")
            # Optional: Add delay between iterations?
            # time.sleep(config.get('iteration_delay', 0))

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting refinement loop.")
            break
        except Exception as e:
            logger.error(f"Error during prompt refinement loop iteration {iteration}: {e}", exc_info=True)
            # Decide whether to break or try to continue
            logger.error("Attempting to continue to the next iteration after delay...")
            time.sleep(10) # Delay before potentially trying again

    logger.info("========== Prompt Refinement Loop Finished ==========")


if __name__ == "__main__":
    # Setup logger - moved inside main or called globally?
    # logger = setup_logger(CONFIG['paths']['logs_dir'], CONFIG) # Ensure logger is set up
    try:
        main(CONFIG)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
        sys.exit(1)
