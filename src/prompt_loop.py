"""
Prompt Loop Module

This module contains the core logic for executing a single iteration of the prompt refinement loop.
It handles the complete workflow for one iteration including:
- Processing input phrases in batches
- Generating paraphrases using the current prompt
- Classifying paraphrases as human or machine
- Filtering and saving selected paraphrases
- Calculating iteration metrics and statistics
- Refining the generator prompt for the next iteration
"""

import sys
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from google import genai

from .config import CONFIG, refine_generator_prompt
from .utils import (
    logger,
    ensure_directory,
    postprocess_discriminator_output_gemini
)
from .provider_facade import generate as provider_generate, classify as provider_classify

def run_prompt_refinement_iteration(
    iteration: int,
    current_generator_prompt: str,
    genai_client: Optional[genai.Client],
    input_data: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """
    Runs a single iteration of the prompt refinement loop using Gemini API.

    Args:
        iteration: The current loop iteration number.
        current_generator_prompt: The prompt template to use for generation.
        gemini_model: The initialized Gemini model object.
        input_data: DataFrame containing 'input_text' column.
        config: The main configuration dictionary.

    Returns:
        A tuple containing:
        - results: A dictionary with metrics and paths from the iteration.
        - next_generator_prompt: The refined prompt for the next iteration.
    """
    logger.info(f"--- Starting Prompt Refinement Iteration {iteration} ---")
    paths = config['paths']
    filenames = config['filenames']
    gemini_config = config['gemini']
    # model_name resolved by provider facade; keep template sources in config
    loop_config = config['loop_control']

    # --- Prepare for Iteration ---
    iteration_results_data: List[Dict[str, str]] = [] # Store detailed results for each input phrase
    processed_count = 0
    generation_success_count = 0
    classification_counts = {'human': 0, 'machine': 0, 'error': 0, 'failed': 0, 'generation_failed': 0}

    # --- Process Input Data in Batches ---
    num_inputs = len(input_data)
    batch_size = loop_config['batch_size']
    sleep_time = loop_config['sleep_between_batches']

    for i in range(0, num_inputs, batch_size):
        batch_df = input_data.iloc[i:min(i + batch_size, num_inputs)]
        logger.info(f"Processing batch {i // batch_size + 1}/{(num_inputs + batch_size - 1) // batch_size} (indices {i}-{min(i + batch_size, num_inputs)-1})...")

        for index, row in batch_df.iterrows():
            input_text = row['input_text']
            processed_count += 1
            logger.debug(f"Processing input: '{input_text}'")

            # 1. Generate Paraphrase via provider facade
            generated_text = provider_generate(
                text=input_text,
                prompt_template=current_generator_prompt,
                client=genai_client
            )

            result_entry = {
                'input_text': input_text,
                'generated_text': None,
                'classification': 'generation_failed'
            }

            if generated_text:
                generation_success_count += 1
                result_entry['generated_text'] = generated_text
                logger.debug(f"Generated: '{generated_text}'")

                # 2. Classify Paraphrase via provider facade
                classification = provider_classify(
                    text=generated_text,
                    classification_prompt_template=gemini_config['classification_prompt_template'],
                    client=genai_client
                )

                if classification:
                    result_entry['classification'] = classification
                    classification_counts[classification] = classification_counts.get(classification, 0) + 1
                    logger.debug(f"Classified as: {classification}")
                else:
                    result_entry['classification'] = 'classification_failed'
                    classification_counts['failed'] = classification_counts.get('failed', 0) + 1
                    logger.warning(f"Classification failed for generated text: '{generated_text}'")
            else:
                logger.warning(f"Generation failed for input: '{input_text}'")
                classification_counts['generation_failed'] = classification_counts.get('generation_failed', 0) + 1
                # Keep classification as 'generation_failed'

            iteration_results_data.append(result_entry)

        # Sleep between batches if configured
        if sleep_time > 0 and i + batch_size < num_inputs:
            logger.debug(f"Sleeping for {sleep_time}s before next batch...")
            time.sleep(sleep_time)

    logger.info("Finished processing all batches.")

    # --- 3. Post-process and Save Selected ---
    logger.info("Post-processing results...")
    selected_pairs_data = postprocess_discriminator_output_gemini(iteration_results_data)
    selected_df = pd.DataFrame(selected_pairs_data)

    # Save selected paraphrases
    selected_output_dir = paths['selected_paraphrases_dir']
    ensure_directory(selected_output_dir)
    selected_filename = filenames['selected_paraphrases'].format(iteration=iteration)
    selected_filepath = selected_output_dir / selected_filename
    try:
        selected_df.to_csv(selected_filepath, sep='\t', index=False, columns=['input_text', 'target_text']) # Save only input/target
        logger.info(f"Saved {len(selected_df)} selected paraphrases to {selected_filepath}")
    except Exception as e:
        logger.error(f"Failed to save selected paraphrases: {e}")

    # --- 4. Calculate Metrics ---
    total_processed = processed_count
    total_generated = generation_success_count
    total_classified_human = classification_counts['human']
    generation_rate = total_generated / total_processed if total_processed > 0 else 0
    selection_rate = total_classified_human / total_generated if total_generated > 0 else 0 # Rate of human classification among generated

    results_summary = {
        "iteration": iteration,
        "total_processed": total_processed,
        "total_generated": total_generated,
        "generation_rate": f"{generation_rate:.2%}",
        "classification_counts": classification_counts,
        "total_selected_human": total_classified_human,
        "selection_rate_of_generated": f"{selection_rate:.2%}",
        "selected_paraphrases_file": str(selected_filepath),
        "generator_prompt_used": current_generator_prompt,
    }
    logger.info(f"Iteration {iteration} Summary: {results_summary}")

    # Save iteration results summary
    results_output_dir = paths['processed_data_dir'] # Save summary in main processed dir
    ensure_directory(results_output_dir)
    results_filename = filenames['loop_results'].format(iteration=iteration)
    results_filepath = results_output_dir / results_filename
    try:
        with open(results_filepath, 'w') as f:
            json.dump(results_summary, f, indent=4)
        logger.info(f"Saved iteration results summary to {results_filepath}")
    except Exception as e:
        logger.error(f"Failed to save iteration results summary: {e}")


    # --- 5. Refine Prompt ---
    logger.info("Refining generator prompt for next iteration...")
    next_generator_prompt = refine_generator_prompt(
        current_prompt=current_generator_prompt,
        results=results_summary, # Pass summary for refinement logic
        strategy=loop_config['prompt_refinement_strategy']
    )

    # Save the prompt used for this iteration and the one for the next
    prompt_history_dir = paths['prompt_history_dir']
    ensure_directory(prompt_history_dir)
    current_prompt_filename = filenames['prompt_history'].format(iteration=iteration)
    current_prompt_filepath = prompt_history_dir / current_prompt_filename
    try:
        with open(current_prompt_filepath, 'w') as f:
            f.write(f"# Prompt used for Iteration {iteration}\n\n")
            f.write(current_generator_prompt)
        logger.info(f"Saved generator prompt for iteration {iteration} to {current_prompt_filepath}")
    except Exception as e:
        logger.error(f"Failed to save current generator prompt: {e}")

    logger.info(f"--- Completed Prompt Refinement Iteration {iteration} ---")
    return results_summary, next_generator_prompt