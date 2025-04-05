import sys
import time
from typing import Dict, Any

from transformers import AutoTokenizer
from datasets import load_dataset

# Import components from the new modules
from .config import CONFIG
from .utils import logger, ensure_directory
from .data_utils import generate_mock_paraphrases
from .model_utils import load_generator_model, load_discriminator_model
from .gan import gan_prep_hf, gan_loop_iteration_hf

# --- Initialize Tokenizers ---
# Use try-except blocks for robustness
try:
    logger.info(f"Loading generator tokenizer: {CONFIG['model_identifiers']['generator']}")
    generator_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_identifiers']['generator'])
except Exception as e:
    logger.critical(f"Failed to load generator tokenizer: {e}")
    sys.exit(1)

try:
    logger.info(f"Loading discriminator tokenizer: {CONFIG['model_identifiers']['discriminator']}")
    discriminator_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_identifiers']['discriminator'])
except Exception as e:
    logger.critical(f"Failed to load discriminator tokenizer: {e}")
    sys.exit(1)


# --- Main Script Execution ---
def main(config: Dict[str, Any]):
    """Main function to execute the GAN workflow."""
    logger.info("========== Starting GAN Paraphrase Generation ==========")

    # Create essential directory structure from config
    paths = config['paths']
    filenames = config['filenames'] # Get filenames config
    ensure_directory(paths['logs_dir'])
    ensure_directory(paths['raw_data_dir'])
    ensure_directory(paths['processed_data_dir'])
    ensure_directory(paths['generator_models'])
    ensure_directory(paths['discriminator_models'])
    ensure_directory(paths['generator_processed_data'])
    ensure_directory(paths['discriminator_processed_data'])
    ensure_directory(paths['generator_generated_output']) # Ensure loop output dirs exist
    ensure_directory(paths['discriminator_classified_output'])
    # Initial/Latest model dirs will be created by training/copying

    # Generate initial mock data (if raw data files don't exist)
    raw_gen_input_path = paths['raw_data_dir'] / filenames['mock_generator_input']
    raw_disc_input_path = paths['raw_data_dir'] / filenames['mock_discriminator_input']
    if not raw_gen_input_path.exists() or not raw_disc_input_path.exists():
        logger.info("Raw data files not found. Generating mock data...")
        try:
            # Use generate_mock_paraphrases from data_utils
            generate_mock_paraphrases(paths['raw_data_dir'], config)
        except Exception as e:
            logger.critical(f"Failed to generate mock data. Cannot proceed. Error: {e}")
            sys.exit(1)
    else:
        logger.info("Raw data files found. Skipping mock data generation.")


    # --- Load and Preprocess Data using Hugging Face datasets ---
    logger.info("Loading and preprocessing data...")
    try:
        # Load the mock TSV files
        gen_data_files = {"train": str(raw_gen_input_path)}
        raw_gen_datasets = load_dataset("csv", data_files=gen_data_files, delimiter="\t")
        disc_data_files = {"train": str(raw_disc_input_path)}
        raw_disc_datasets = load_dataset("csv", data_files=disc_data_files, delimiter="\t")

        # --- Define Preprocessing Functions (kept here for clarity, could move) ---
        max_input_length = config['training']['generator_max_length']
        max_target_length = config['training']['generator_max_length']
        discriminator_max_length = config['training']['discriminator_max_length']

        def preprocess_generator(examples):
            """Tokenizes generator data (input and target phrases)."""
            inputs = examples["input_phrase"]
            targets = examples["target_phrase"]
            inputs = ["" if i is None else i for i in inputs]
            targets = ["" if t is None else t for t in targets]
            model_inputs = generator_tokenizer(inputs, max_length=max_input_length, truncation=True)
            with generator_tokenizer.as_target_tokenizer():
                labels = generator_tokenizer(targets, max_length=max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def preprocess_discriminator(examples):
            """Tokenizes discriminator data (phrases)."""
            phrases = examples["phrase"]
            phrases = ["" if p is None else p for p in phrases]
            tokenized_inputs = discriminator_tokenizer(phrases, truncation=True, max_length=discriminator_max_length)
            tokenized_inputs["labels"] = examples["label"] # Pass labels through
            return tokenized_inputs

        # --- Apply Preprocessing ---
        logger.info("Applying preprocessing...")
        tokenized_gen_datasets = raw_gen_datasets.map(preprocess_generator, batched=True, remove_columns=raw_gen_datasets["train"].column_names)
        tokenized_disc_datasets = raw_disc_datasets.map(preprocess_discriminator, batched=True, remove_columns=["phrase"])
        logger.info(f"Generator dataset tokenized: {tokenized_gen_datasets}")
        logger.info(f"Discriminator dataset tokenized: {tokenized_disc_datasets}")

    except Exception as e:
        logger.critical(f"Failed during data loading/preprocessing. Cannot proceed. Error: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Initial Models ---
    logger.info("Loading initial Hugging Face models...")
    # Use load functions from model_utils
    generator_model = load_generator_model(config['model_identifiers']['generator'])
    discriminator_model = load_discriminator_model(config['model_identifiers']['discriminator'])
    logger.info("Initial models loaded.")

    # --- GAN Prep using HF Trainer ---
    logger.info("Starting GAN preparation (initial training)...")
    try:
        # Use gan_prep_hf from gan module
        gan_prep_hf(
            config=config,
            gen_model=generator_model,
            disc_model=discriminator_model,
            gen_tokenizer=generator_tokenizer,
            disc_tokenizer=discriminator_tokenizer,
            tokenized_gen_dataset=tokenized_gen_datasets['train'],
            tokenized_disc_dataset=tokenized_disc_datasets['train']
        )
    except Exception as e:
         logger.critical(f"GAN preparation (initial training) failed. Cannot proceed. Error: {e}", exc_info=True)
         sys.exit(1)


    # Continuous GAN loop
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"========== Starting GAN Iteration {iteration} ==========")
        # --- Call the HF GAN loop iteration ---
        try:
            # Use gan_loop_iteration_hf from gan module
            gan_loop_iteration_hf(
                config=config,
                gen_tokenizer=generator_tokenizer,
                disc_tokenizer=discriminator_tokenizer
            )
            logger.info(f"========== Completed GAN Iteration {iteration} ==========")
            # Optional: Add delay?
            # time.sleep(5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting GAN loop.")
            break
        except Exception as e:
            logger.error(f"Error during GAN loop iteration {iteration}: {e}", exc_info=True)
            logger.error("Attempting to continue to the next iteration after delay...")
            time.sleep(10) # Delay before next attempt


if __name__ == "__main__":
    try:
        main(CONFIG)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
        sys.exit(1)
