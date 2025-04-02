import os
import shutil
import subprocess
import sys
import logging
import time
import itertools
from pathlib import Path
from typing import Dict, Any, Set, List
import numpy as np # For processing predictions

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets # Added concatenate_datasets

# --- Configuration ---

PROJECT_ROOT = Path(__file__).parent.parent

CONFIG: Dict[str, Any] = {
    "paths": {
        "project_root": PROJECT_ROOT,
        "config_dir": PROJECT_ROOT / "config",
        "scripts_dir": PROJECT_ROOT / "scripts",
        "src_dir": PROJECT_ROOT / "src",
        "data_root": PROJECT_ROOT / "data",
        "models_root": PROJECT_ROOT / "models",
        "logs_dir": PROJECT_ROOT / "logs",
        "raw_data_dir": PROJECT_ROOT / "data" / "raw",
        "processed_data_dir": PROJECT_ROOT / "data" / "processed",
        "generator_models": PROJECT_ROOT / "models" / "generator",
        "discriminator_models": PROJECT_ROOT / "models" / "discriminator",
        "generator_initial_model": PROJECT_ROOT / "models" / "generator" / "initial",
        "generator_latest_model": PROJECT_ROOT / "models" / "generator" / "latest",
        "discriminator_initial_model": PROJECT_ROOT / "models" / "discriminator" / "initial",
        "generator_processed_data": PROJECT_ROOT / "data" / "processed" / "generator", # Dir for combined data
        "discriminator_processed_data": PROJECT_ROOT / "data" / "processed" / "discriminator", # Dir for selected data
        "generator_generated_output": PROJECT_ROOT / "data" / "processed" / "generator" / "generated", # Dir for raw generated text
        "discriminator_classified_output": PROJECT_ROOT / "data" / "processed" / "discriminator" / "classified", # Dir for selected generated TSV
    },
    "filenames": {
        "log_file": "run.log",
        # Input files in raw_data_dir (created by mock data generation)
        "mock_generator_input": "mock_generator_input.tsv",
        "mock_discriminator_input": "mock_discriminator_input.tsv",
        # Files generated during GAN loop
        "generator_generated_phrases": "generated_phrases.txt", # Raw output from generator.generate()
        "discriminator_input_generated": "discriminator_input_generated.tsv", # Formatted generated phrases for discriminator input (CSV with header 'phrase')
        "discriminator_selected_generated": "discriminator_selected_generated.tsv", # Generated phrases selected by discriminator (TSV: input_phrase, target_phrase)
        "generator_combined_training_data": "generator_combined_training_data.tsv", # Combined data for next generator training (TSV: input_phrase, target_phrase)
    },
    "model_identifiers": { # Renamed from model_names
        "generator": "t5-small",
        "discriminator": "bert-base-uncased",
    },
    # Removed problem_names as they are trax specific
    # Removed trax specific hparams, will add HF TrainingArguments later
    "training": {
        # Adjusted training params for HF Trainer (epochs are more common than steps)
        "num_train_epochs": 1, # Example: train for 1 epoch initially
        "per_device_train_batch_size": 8, # Example batch size
        "per_device_eval_batch_size": 8, # Example batch size
        "logging_steps": 100, # Log every 100 steps
        "eval_steps": 100, # Evaluate every 100 steps (can be evaluation_strategy='steps')
        "save_steps": 500, # Save checkpoint every 500 steps
        # "decode_steps": 100, # Not directly used by HF generate/predict
        "gan_prep_max_retries": 3,
        "gan_prep_retry_delay": 5,
        "mock_data_lines": 500, # Reduced for faster testing/debugging
        "discriminator_decode_steps": 1, # Specific decode steps for discriminator classification - Not directly applicable to HF predict
        "generator_max_length": 128,
        "discriminator_max_length": 128,
        "generator_update_learning_rate": 2e-5, # Potentially smaller LR for updates
    },
    # Removed trax specific constants
    "logging": {
        "logger_name": "gan_paraphrase",
        "console_level": logging.INFO,
        "file_level": logging.DEBUG,
    }
}

# --- Helper Functions ---

def setup_logger(log_dir: Path, config: Dict[str, Any]) -> logging.Logger:
    """Sets up a logger with console and file handlers based on config."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / config['filenames']['log_file']
    logger = logging.getLogger(config['logging']['logger_name'])
    logger.setLevel(logging.DEBUG) # Set root logger level to lowest

    # Prevent adding handlers multiple times if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(config['logging']['console_level'])
    formatter_ch = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter_ch)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(config['logging']['file_level'])
    formatter_fh = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)

    return logger

# Initialize logger using the config
logger = setup_logger(CONFIG['paths']['logs_dir'], CONFIG)

# --- Initialize Tokenizers ---
# Use try-except blocks for robustness, although dependencies should be installed
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


def run_shell_command(
    command: list[str], # Use list for better security than shell=True
    cwd: str = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Runs a shell command with optional working directory and output capture.
    Provides more consistent error handling. Avoids shell=True.
    """
    try:
        logger.debug(f"Running command: {' '.join(command)} in {cwd or os.getcwd()}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
            shell=False, # Avoid shell=True for security
        )
        if capture_output:
             logger.debug(f"Command stdout:\n{result.stdout}")
             logger.debug(f"Command stderr:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        raise  # Re-raise the exception to stop execution
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        raise


def ensure_directory(path: Path):
    """Creates a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


# --- Data Preparation ---
# Keeping DataPreparer for now, although mock data generation bypasses it.
# It might be useful if switching to non-mock data later.
class DataPreparer:
    """Prepares data by reading, cleaning, normalizing, and writing TSV files."""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        logger.info(f"DataPreparer initialized: Input={self.input_dir}, Output={self.output_dir}")

    def prepare(self):
        """Prepares the data: validates inputs, cleans output, processes files."""
        logger.info("Starting data preparation...")
        self._validate_inputs()
        self._clean_output()
        self._process_files()
        logger.info("Data preparation complete.")

    def _validate_inputs(self):
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} not found")
            raise FileNotFoundError(f"Input directory {self.input_dir} not found")
        logger.debug("Input directory validated.")

    def _clean_output(self):
        if self.output_dir.exists():
            logger.warning(f"Removing existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        ensure_directory(self.output_dir)
        logger.debug("Output directory cleaned/created.")

    def _process_files(self):
        files_processed = 0
        for file in self.input_dir.glob("*.tsv"):
            self._process_file(file)
            files_processed += 1
        if files_processed == 0:
             logger.warning(f"No .tsv files found in {self.input_dir}")
        else:
             logger.info(f"Processed {files_processed} TSV files.")


    def _process_file(self, file_path: Path):
        output_file = self.output_dir / file_path.name
        lines_written = 0
        lines_read = 0
        logger.debug(f"Processing file: {file_path} -> {output_file}")
        try:
            with open(file_path, "r", encoding="utf-8") as infile, open(
                output_file, "w", encoding="utf-8"
            ) as outfile:
                for line in infile:
                    lines_read += 1
                    cleaned = self._clean_line(line)
                    if cleaned:
                        outfile.write(f"{cleaned}\n")
                        lines_written += 1
            logger.info(f"Processed {file_path.name}: Read {lines_read} lines, wrote {lines_written} cleaned lines.")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    @staticmethod
    def _clean_line(line: str) -> str:
        """Cleans and normalizes a single line of text."""
        line = line.strip().lower()
        # Basic check for TSV format (at least one tab -> two columns)
        if not line or '\t' not in line:
            logger.debug(f"Skipping invalid/empty line: '{line[:50]}...'")
            return ""
        return line


# --- Removed Trax Problem Definitions and Model Definitions ---
# These will be replaced by Hugging Face datasets, tokenizers, and models.


# --- Removed Trax Training and Decoding functions ---
# Will be replaced by HF Trainer and model.generate()/predict()
# --- Hugging Face Model Loading ---

def load_generator_model(model_identifier: str):
    """Loads the generator model (Seq2Seq)."""
    logger.info(f"Loading generator model: {model_identifier}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)
        return model
    except Exception as e:
        logger.critical(f"Failed to load generator model '{model_identifier}': {e}", exc_info=True)
        sys.exit(1)

def load_discriminator_model(model_identifier: str, num_labels: int = 2):
    """Loads the discriminator model (Sequence Classification)."""
    logger.info(f"Loading discriminator model: {model_identifier}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=num_labels)
        return model
    except Exception as e:
        logger.critical(f"Failed to load discriminator model '{model_identifier}': {e}", exc_info=True)
        sys.exit(1)


# --- Hugging Face Training ---

def train_hf_model(
    model: Any, # Can be Seq2Seq or SequenceClassification model
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset = None, # Optional eval dataset
    training_args: TrainingArguments,
    data_collator: Any, # DataCollatorForSeq2Seq or DataCollatorWithPadding
):
    """Trains a Hugging Face model using the Trainer API."""
    logger.info(f"Initializing Trainer for model: {model.__class__.__name__}")
    logger.info(f"Training Arguments: {training_args}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        logger.info("Starting training...")
        train_result = trainer.train()
        logger.info("Training complete.")
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # Save model and tokenizer
        trainer.save_model() # Saves the model to training_args.output_dir
        # Save tokenizer explicitly as well, although Trainer might do it sometimes
        if tokenizer:
             tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Model and tokenizer saved to {training_args.output_dir}")
        return trainer # Return trainer for potential further use

    except Exception as e:
        logger.error(f"Training failed for {model.__class__.__name__}: {e}", exc_info=True)
        raise


# --- Main GAN Workflow Scripts ---

# Insert gan_prep_hf definition here
def gan_prep_hf(
    config: Dict[str, Any],
    gen_model: AutoModelForSeq2SeqLM,
    disc_model: AutoModelForSequenceClassification,
    gen_tokenizer: AutoTokenizer,
    disc_tokenizer: AutoTokenizer,
    tokenized_gen_dataset: Dataset,
    tokenized_disc_dataset: Dataset,
):
    """Prepares the GAN by training initial HF generator and discriminator models."""
    logger.info("--- Starting GAN Preparation (Hugging Face) ---")
    paths = config['paths']
    training_config = config['training']

    # --- Define Training Arguments ---
    # Generator Training Args
    gen_output_dir = paths['generator_initial_model']
    ensure_directory(gen_output_dir)
    gen_training_args = TrainingArguments(
        output_dir=str(gen_output_dir),
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        logging_dir=str(paths['logs_dir'] / "gen_initial_train"),
        logging_steps=training_config['logging_steps'],
        # evaluation_strategy="steps", # Add if eval dataset is provided
        # eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=2, # Keep only last 2 checkpoints
        report_to="none", # Disable wandb/tensorboard for now
        # Add other relevant args: learning_rate, weight_decay, warmup_steps etc.
        learning_rate=5e-5, # Example LR
        predict_with_generate=True, # Important for Seq2Seq models
    )

    # Discriminator Training Args
    disc_output_dir = paths['discriminator_initial_model']
    ensure_directory(disc_output_dir)
    disc_training_args = TrainingArguments(
        output_dir=str(disc_output_dir),
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        logging_dir=str(paths['logs_dir'] / "disc_initial_train"),
        logging_steps=training_config['logging_steps'],
        # evaluation_strategy="steps", # Add if eval dataset is provided
        # eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        save_total_limit=2,
        report_to="none",
        learning_rate=5e-5, # Example LR
    )

    # --- Define Data Collators ---
    gen_data_collator = DataCollatorForSeq2Seq(tokenizer=gen_tokenizer, model=gen_model)
    disc_data_collator = DataCollatorWithPadding(tokenizer=disc_tokenizer)

    # --- Train Initial Generator ---
    logger.info("Training initial generator...")
    try:
        train_hf_model(
            model=gen_model,
            tokenizer=gen_tokenizer,
            train_dataset=tokenized_gen_dataset, # Assumes 'train' split exists
            # eval_dataset=tokenized_gen_dataset['validation'], # Add if split exists
            training_args=gen_training_args,
            data_collator=gen_data_collator,
        )
        # Copy initial model to latest for the first loop iteration
        latest_gen_path = paths['generator_latest_model']
        if latest_gen_path.exists():
            logger.warning(f"Removing existing latest generator model at: {latest_gen_path}")
            shutil.rmtree(latest_gen_path)
        # The train_hf_model function saved the trained model to gen_output_dir.
        # Now, ensure the 'latest' directory exists and save the trained model there.
        ensure_directory(latest_gen_path)
        # Load the model that was just trained and saved by the Trainer to gen_output_dir
        # Note: Trainer modifies the model in-place, so we can save gen_model directly
        gen_model.save_pretrained(latest_gen_path)
        # Also save the corresponding tokenizer to the latest path
        gen_tokenizer.save_pretrained(latest_gen_path)
        logger.info(f"Saved initially trained generator model and tokenizer to {latest_gen_path}")

    except Exception as e:
        logger.error(f"GAN Prep failed during initial generator training: {e}", exc_info=True)
        sys.exit(1) # Exit if initial training fails

    # --- Train Initial Discriminator ---
    logger.info("Training initial discriminator...")
    try:
        train_hf_model(
            model=disc_model,
            tokenizer=disc_tokenizer,
            train_dataset=tokenized_disc_dataset, # Assumes 'train' split exists
            # eval_dataset=tokenized_disc_dataset['validation'], # Add if split exists
            training_args=disc_training_args,
            data_collator=disc_data_collator,
        )
        # Discriminator model is saved to its initial dir by train_hf_model. No copy needed.
    except Exception as e:
        logger.error(f"GAN Prep failed during initial discriminator training: {e}", exc_info=True)
        sys.exit(1) # Exit if initial training fails

    logger.info("--- GAN Preparation Complete (Hugging Face) ---")


# Insert gan_loop_iteration_hf definition here
def gan_loop_iteration_hf(
    config: Dict[str, Any],
    # Pass necessary components if not loaded globally/within function
    # gen_tokenizer: AutoTokenizer, # Assuming global access for now
    # disc_tokenizer: AutoTokenizer, # Assuming global access for now
):
    """Runs a single iteration of the GAN training loop using HF."""
    logger.info("--- Starting GAN Loop Iteration (Hugging Face) ---")
    paths = config['paths']
    training_config = config['training']
    filenames = config['filenames']

    # Define paths for this iteration (using config for clarity)
    latest_gen_model_path = paths['generator_latest_model']
    initial_disc_model_path = paths['discriminator_initial_model'] # Using initial discriminator for now
    gen_output_dir = paths['generator_generated_output'] # Where generated phrases go (as text file)
    disc_classified_dir = paths['discriminator_classified_output'] # Where selected phrases go
    raw_gen_input_path = paths['raw_data_dir'] / filenames['mock_generator_input'] # Input for generation
    original_gen_data_path = paths['raw_data_dir'] / filenames['mock_generator_input'] # Base data to combine with later
    generated_phrases_raw_file = gen_output_dir / filenames['generator_generated_phrases']
    discriminator_input_generated_file = gen_output_dir / filenames['discriminator_input_generated'] # TSV with 'phrase' header
    discriminator_selected_file = disc_classified_dir / filenames['discriminator_selected_generated']
    combined_training_file = paths['generator_processed_data'] / filenames['generator_combined_training_data']


    ensure_directory(gen_output_dir)
    ensure_directory(disc_classified_dir)
    ensure_directory(paths['generator_processed_data']) # Ensure combined data dir exists

    # --- 1. Load latest generator model ---
    logger.info(f"Loading latest generator model from: {latest_gen_model_path}")
    try:
        # Assuming generator_tokenizer is loaded globally
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(str(latest_gen_model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_model.to(device)
    except Exception as e:
        logger.error(f"Failed to load latest generator model: {e}", exc_info=True)
        raise

    # --- 2. Prepare input data for generation ---
    logger.info(f"Loading generation input data from: {raw_gen_input_path}")
    try:
        # Load only the input phrases for generation
        gen_input_dataset = load_dataset("csv", data_files={"predict": str(raw_gen_input_path)}, delimiter="\t")['predict']
        # Store original phrases for pairing later in postprocessing
        original_input_phrases = list(gen_input_dataset['input_phrase'])
        unique_input_phrases = sorted(list(set(original_input_phrases))) # Sort for consistent order

        logger.info(f"Loaded {len(unique_input_phrases)} unique phrases for generation.")
        # Prepare dataset for prediction (just the unique input phrases)
        generation_input_ds = Dataset.from_dict({"input_phrase": unique_input_phrases})

        def tokenize_gen_input(examples):
             # T5 prefix is already in the mock data
             return generator_tokenizer(examples["input_phrase"], padding=False, truncation=True, max_length=training_config['generator_max_length']) # No padding needed for generate

        tokenized_generation_input_ds = generation_input_ds.map(tokenize_gen_input, batched=True, remove_columns=["input_phrase"])
        tokenized_generation_input_ds.set_format("torch") # Ensure tensors are PyTorch tensors

    except Exception as e:
        logger.error(f"Failed to prepare generation input data: {e}", exc_info=True)
        raise

    # --- 3. Generate Paraphrases ---
    logger.info("Generating paraphrases...")
    generated_phrase_texts: List[str] = []
    try:
        gen_model.eval() # Set model to evaluation mode
        with torch.no_grad():
             outputs = gen_model.generate(
                 input_ids=tokenized_generation_input_ds["input_ids"].to(device),
                 attention_mask=tokenized_generation_input_ds["attention_mask"].to(device),
                 max_length=training_config['generator_max_length'], # Max length for generated output
                 num_beams=4, # Example beam search
                 early_stopping=True
             )
        generated_phrase_texts = generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info(f"Generated {len(generated_phrase_texts)} paraphrases for {len(unique_input_phrases)} unique inputs.")

        # Save generated phrases to a file for discriminator input (TSV with header)
        with open(discriminator_input_generated_file, "w", encoding="utf-8") as f_out:
             f_out.write("phrase\n") # Header for discriminator dataset loading
             for phrase in generated_phrase_texts:
                 # Basic cleaning
                 phrase_clean = phrase.replace('\t', ' ').replace('\n', ' ').strip()
                 if phrase_clean: # Avoid writing empty lines
                    f_out.write(f"{phrase_clean}\n")
        logger.info(f"Saved generated phrases for discriminator input to {discriminator_input_generated_file}")

    except Exception as e:
        logger.error(f"Failed during paraphrase generation: {e}", exc_info=True)
        raise

    # --- 4 & 5. Load Discriminator and Prepare Generated Data ---
    logger.info(f"Loading initial discriminator model from: {initial_disc_model_path}")
    try:
        disc_model = AutoModelForSequenceClassification.from_pretrained(str(initial_disc_model_path))
        # Assuming discriminator_tokenizer is loaded globally
        disc_model.to(device)

        # Load the generated phrases file into a dataset
        generated_ds = load_dataset("csv", data_files={"predict": str(discriminator_input_generated_file)}, delimiter="\t")['predict']

        def tokenize_disc_input(examples):
             phrases = examples["phrase"]
             phrases = ["" if p is None else p for p in phrases] # Handle potential None
             return discriminator_tokenizer(phrases, padding="max_length", truncation=True, max_length=training_config['discriminator_max_length'])

        tokenized_generated_ds = generated_ds.map(tokenize_disc_input, batched=True, remove_columns=["phrase"])
        tokenized_generated_ds.set_format("torch") # Ensure tensors for prediction

    except Exception as e:
        logger.error(f"Failed to load discriminator or prepare generated data: {e}", exc_info=True)
        raise

    # --- 6. Classify Generated Phrases ---
    logger.info("Classifying generated phrases...")
    predictions = None
    predicted_labels = None
    try:
        # Use Trainer for prediction
        predict_args = TrainingArguments(
            output_dir=str(disc_classified_dir / "predict_temp"), # Temporary dir
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            report_to="none",
            do_train=False,
            do_predict=True,
            remove_unused_columns=False, # Keep potential extra columns if any
        )
        disc_data_collator = DataCollatorWithPadding(tokenizer=discriminator_tokenizer)
        trainer = Trainer(
            model=disc_model,
            args=predict_args,
            tokenizer=discriminator_tokenizer,
            data_collator=disc_data_collator,
        )
        predictions = trainer.predict(tokenized_generated_ds)
        predicted_logits = predictions.predictions
        predicted_labels = np.argmax(predicted_logits, axis=-1) # Get label index (0 or 1)
        logger.info(f"Classification complete. Found {np.sum(predicted_labels == 1)} potential 'human-like' phrases.")

    except Exception as e:
        logger.error(f"Failed during phrase classification: {e}", exc_info=True)
        raise

    # --- 7. Post-process Discriminator Output ---
    logger.info("Post-processing discriminator output...")
    try:
        # Pass the original unique input phrases and the generated phrases along with predictions
        postprocess_discriminator_output_hf(
            predictions=predicted_labels, # Numpy array of 0s and 1s
            unique_input_phrases=unique_input_phrases, # List of original inputs used for generation
            generated_phrases=generated_phrase_texts, # List of corresponding generated outputs
            output_file=discriminator_selected_file,
            human_label_index=1 # Assuming label '1' is human
        )
    except Exception as e:
        logger.error(f"Failed Step 7 (Post-process Discriminator Output): {e}", exc_info=True)
        raise

    # --- 8. Update Generator Training Data ---
    logger.info("Updating Generator Training Data...")
    try:
        combine_data_hf(
            original_data_path=original_gen_data_path, # Path to original human data TSV
            selected_generated_path=discriminator_selected_file, # Path to newly selected generated data TSV
            output_path=combined_training_file,
        )
    except Exception as e:
        logger.error(f"Failed Step 8 (Combine Data): {e}", exc_info=True)
        raise

    # --- 9. Train Generator (Update Latest) ---
    logger.info("Retraining Generator...")
    try:
        # Load the combined dataset
        combined_dataset = load_dataset("csv", data_files={"train": str(combined_training_file)}, delimiter="\t")['train']

        # Define Preprocessing function (same as initial prep)
        def preprocess_generator_update(examples):
            inputs = examples["input_phrase"]
            targets = examples["target_phrase"]
            inputs = ["" if i is None else i for i in inputs]
            targets = ["" if t is None else t for t in targets]
            model_inputs = generator_tokenizer(inputs, max_length=training_config['generator_max_length'], truncation=True)
            with generator_tokenizer.as_target_tokenizer():
                labels = generator_tokenizer(targets, max_length=training_config['generator_max_length'], truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_combined_dataset = combined_dataset.map(
            preprocess_generator_update,
            batched=True,
            remove_columns=combined_dataset.column_names
        )

        # Define Training Arguments for the update step
        gen_update_args = TrainingArguments(
            output_dir=str(latest_gen_model_path), # Save directly to the 'latest' path
            num_train_epochs=training_config['num_train_epochs'], # Or adjust epochs for update
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            logging_dir=str(paths['logs_dir'] / "gen_update_train"),
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=2,
            report_to="none",
            learning_rate=training_config.get('generator_update_learning_rate', 2e-5), # Use specific LR if defined
            predict_with_generate=True,
            overwrite_output_dir=True # Overwrite the previous 'latest' model
        )
        gen_data_collator = DataCollatorForSeq2Seq(tokenizer=generator_tokenizer, model=gen_model)

        # Train the generator model (loaded at the start of the loop)
        train_hf_model(
            model=gen_model,
            tokenizer=generator_tokenizer,
            train_dataset=tokenized_combined_dataset,
            training_args=gen_update_args, # Use specific args for update
            data_collator=gen_data_collator,
        )
        # Model and tokenizer are saved within train_hf_model to latest_gen_model_path

    except Exception as e:
        logger.error(f"Failed Step 9 (Train Generator): {e}", exc_info=True)
        raise # Stop iteration if training fails

    # --- 10. (Optional) Train Discriminator ---
    # TODO: Implement discriminator update logic if desired

    logger.info("--- GAN Loop Iteration Complete ---")


# Rewrite postprocess_discriminator_output
def postprocess_discriminator_output_hf(
    predictions: np.ndarray,
    unique_input_phrases: List[str],
    generated_phrases: List[str],
    output_file: Path,
    human_label_index: int = 1
):
    """Filters generated paraphrases based on discriminator predictions and saves them."""
    logger.info(f"Post-processing HF predictions. Saving selected to '{output_file}'")
    lines_written = 0
    selected_pairs = []

    if len(predictions) != len(generated_phrases) or len(generated_phrases) != len(unique_input_phrases):
        logger.error(f"Mismatch in lengths: predictions ({len(predictions)}), generated ({len(generated_phrases)}), unique inputs ({len(unique_input_phrases)})")
        # Handle error appropriately, maybe raise an exception
        return

    for i, label in enumerate(predictions):
        if label == human_label_index:
            # Pair the selected generated phrase with its original input phrase
            original_input = unique_input_phrases[i]
            selected_generated = generated_phrases[i]
            # Ensure the original input has the T5 prefix if needed for consistency
            # (Mock data already includes it, but good practice for real data)
            t5_prefix = "paraphrase: "
            if not original_input.startswith(t5_prefix):
                 original_input_with_prefix = f"{t5_prefix}{original_input}"
            else:
                 original_input_with_prefix = original_input

            selected_pairs.append((original_input_with_prefix, selected_generated))
            lines_written += 1

    try:
        ensure_directory(output_file.parent) # Ensure output directory exists
        with open(output_file, "w", encoding="utf-8") as of:
            of.write("input_phrase\ttarget_phrase\n") # Header
            for inp_phrase, gen_phrase in selected_pairs:
                 # Basic cleaning
                 inp_clean = inp_phrase.replace('\t', ' ').replace('\n', ' ').strip()
                 gen_clean = gen_phrase.replace('\t', ' ').replace('\n', ' ').strip()
                 if inp_clean and gen_clean: # Ensure neither is empty after cleaning
                     of.write(f"{inp_clean}\t{gen_clean}\n")

        logger.info(f"Post-processing complete. Selected {lines_written} generated phrases.")
    except Exception as e:
        logger.error(f"Error writing selected generated phrases to {output_file}: {e}")
        raise


# Rewrite combine_data
def combine_data_hf(original_data_path: Path, selected_generated_path: Path, output_path: Path):
    """Combines original generator training data with selected generated data."""
    logger.info(f"Combining data: '{original_data_path.name}' + '{selected_generated_path.name}' -> '{output_path.name}'")
    combined_lines: Set[str] = set()
    files_read = 0
    lines_added_from_original = 0
    lines_added_from_generated = 0

    # Read original data
    if original_data_path.exists():
        files_read += 1
        try:
            with open(original_data_path, "r", encoding="utf-8") as f:
                header = next(f).strip() # Read header
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        if cleaned_line not in combined_lines:
                            combined_lines.add(cleaned_line)
                            lines_added_from_original += 1
            logger.debug(f"Read {lines_added_from_original} unique lines from {original_data_path.name}")
        except Exception as e:
             logger.error(f"Error reading file {original_data_path} during combination: {e}")
             # Decide if error is fatal
             raise
    else:
        logger.error(f"Combine data: Original data file not found: {original_data_path}")
        raise FileNotFoundError(f"Original data file not found: {original_data_path}")

    # Read selected generated data
    if selected_generated_path.exists():
        files_read += 1
        try:
            with open(selected_generated_path, "r", encoding="utf-8") as f:
                header = next(f).strip() # Read header
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        if cleaned_line not in combined_lines:
                            combined_lines.add(cleaned_line)
                            lines_added_from_generated += 1
            logger.debug(f"Read {lines_added_from_generated} unique lines from {selected_generated_path.name}")
        except Exception as e:
             logger.error(f"Error reading file {selected_generated_path} during combination: {e}")
             # Non-fatal, maybe just log warning
             logger.warning(f"Could not read selected generated data from {selected_generated_path}. Proceeding without it.")
    else:
        logger.warning(f"Combine data: Selected generated file does not exist, skipping: {selected_generated_path}")

    if not combined_lines:
         logger.warning("No lines found in input files to combine. Output file will be empty.")

    # Write unique lines to the output file
    try:
        ensure_directory(output_path.parent) # Ensure output directory exists
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write("input_phrase\ttarget_phrase\n") # Write header
            # Sort for consistency
            for line in sorted(list(combined_lines)):
                outfile.write(line + "\n")
        total_lines_written = len(combined_lines)
        logger.info(f"Combine data complete. Read {files_read} files, wrote {total_lines_written} unique lines to {output_path.name}")
    except Exception as e:
        logger.error(f"Error writing combined data to {output_path}: {e}")
        raise


# --- Mock Data Generation ---

def generate_mock_paraphrases(output_dir: Path, config: Dict[str, Any]): # Keep signature simple for now
    """Generates mock paraphrase data based on config."""
    ensure_directory(output_dir)
    n_lines = config['training']['mock_data_lines']
    filenames = config['filenames']
    # Define filenames for generator input and discriminator input
    gen_input_file = output_dir / filenames['mock_generator_input'] # Use specific names for clarity
    disc_input_file = output_dir / filenames['mock_discriminator_input']
    logger.info(f"Generating mock data ({n_lines} lines) in {output_dir}...")

    try:
        # Mock data for Generator Training (input_phrase, target_phrase)
        # T5 requires a prefix, e.g., "paraphrase: "
        t5_prefix = "paraphrase: "
        with open(gen_input_file, "w", encoding="utf-8") as f_gen_in:
            f_gen_in.write("input_phrase\ttarget_phrase\n") # Header
            for i in range(n_lines):
                phrase1 = f"human phrase {i+1}"
                paraphrase1a = f"human paraphrase {i+1}a"
                paraphrase1b = f"human paraphrase {i+1}b"
                # Write pairs for training
                f_gen_in.write(f"{t5_prefix}{phrase1}\t{paraphrase1a}\n")
                f_gen_in.write(f"{t5_prefix}{paraphrase1a}\t{phrase1}\n") # Reverse pair
                f_gen_in.write(f"{t5_prefix}{phrase1}\t{paraphrase1b}\n")
                f_gen_in.write(f"{t5_prefix}{paraphrase1b}\t{phrase1}\n") # Reverse pair

                if i % 2 == 0:
                    phrase2 = f"another human phrase {i+1}"
                    paraphrase2 = f"another good paraphrase {i+1}"
                    f_gen_in.write(f"{t5_prefix}{phrase2}\t{paraphrase2}\n")
                    f_gen_in.write(f"{t5_prefix}{paraphrase2}\t{phrase2}\n") # Reverse pair

        # Mock data for Initial Discriminator Training (phrase, label)
        # Label 1: human, Label 0: machine (as defined in old Trax problem)
        with open(disc_input_file, "w", encoding="utf-8") as f_disc_in:
            f_disc_in.write("phrase\tlabel\n") # Header
            for i in range(n_lines):
                # Human examples
                f_disc_in.write(f"human phrase {i+1}\t1\n")
                f_disc_in.write(f"human paraphrase {i+1}a\t1\n")
                # Machine examples (simulated)
                f_disc_in.write(f"machine phrase {i+1}x\t0\n")
                f_disc_in.write(f"machine phrase {i+1}y\t0\n")

                if i % 2 == 0:
                     f_disc_in.write(f"another human phrase {i+1}\t1\n")
                     f_disc_in.write(f"another good paraphrase {i+1}\t1\n")
                if i % 3 == 0:
                     f_disc_in.write(f"another machine attempt {i+1}\t0\n")

        logger.info(f"Generated mock files: {gen_input_file.name}, {disc_input_file.name}")
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        raise


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
    # Use the specific mock file names from config
    raw_gen_input_path = paths['raw_data_dir'] / filenames['mock_generator_input']
    raw_disc_input_path = paths['raw_data_dir'] / filenames['mock_discriminator_input']
    if not raw_gen_input_path.exists() or not raw_disc_input_path.exists():
        logger.info("Raw data files not found. Generating mock data...")
        try:
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
        # Generator data
        gen_data_files = {"train": str(raw_gen_input_path)}
        raw_gen_datasets = load_dataset("csv", data_files=gen_data_files, delimiter="\t") # Let datasets infer columns from header
        # Discriminator data
        disc_data_files = {"train": str(raw_disc_input_path)}
        raw_disc_datasets = load_dataset("csv", data_files=disc_data_files, delimiter="\t") # Let datasets infer columns from header

        # --- Define Preprocessing Functions ---
        max_input_length = config['training']['generator_max_length']
        max_target_length = config['training']['generator_max_length'] # Often same as input for paraphrasing
        discriminator_max_length = config['training']['discriminator_max_length']

        def preprocess_generator(examples):
            """Tokenizes generator data (input and target phrases)."""
            # T5 expects inputs like "paraphrase: original phrase"
            # The mock data generation already added this prefix
            inputs = examples["input_phrase"]
            targets = examples["target_phrase"]
            # Handle potential None values if dataset loading had issues
            inputs = ["" if i is None else i for i in inputs]
            targets = ["" if t is None else t for t in targets]

            model_inputs = generator_tokenizer(inputs, max_length=max_input_length, truncation=True)

            # Setup the tokenizer for targets
            with generator_tokenizer.as_target_tokenizer():
                labels = generator_tokenizer(targets, max_length=max_target_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def preprocess_discriminator(examples):
            """Tokenizes discriminator data (phrases)."""
            phrases = examples["phrase"]
            phrases = ["" if p is None else p for p in phrases] # Handle potential None
            # Ensure label column is handled correctly (it should be passed through)
            tokenized_inputs = discriminator_tokenizer(phrases, truncation=True, max_length=discriminator_max_length)
            # Add labels back if they were removed by tokenizer (shouldn't happen with just text input)
            tokenized_inputs["labels"] = examples["label"] # Pass labels through
            return tokenized_inputs

        # --- Apply Preprocessing ---
        logger.info("Applying preprocessing...")
        tokenized_gen_datasets = raw_gen_datasets.map(preprocess_generator, batched=True, remove_columns=raw_gen_datasets["train"].column_names)
        # For discriminator, keep the 'label' column
        tokenized_disc_datasets = raw_disc_datasets.map(preprocess_discriminator, batched=True, remove_columns=["phrase"])
        logger.info(f"Generator dataset tokenized: {tokenized_gen_datasets}")
        logger.info(f"Discriminator dataset tokenized: {tokenized_disc_datasets}")

    except Exception as e:
        logger.critical(f"Failed during data loading/preprocessing. Cannot proceed. Error: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Initial Models ---
    logger.info("Loading initial Hugging Face models...")
    generator_model = load_generator_model(config['model_identifiers']['generator'])
    discriminator_model = load_discriminator_model(config['model_identifiers']['discriminator'])
    logger.info("Initial models loaded.")

    # --- GAN Prep using HF Trainer ---
    logger.info("Starting GAN preparation (initial training)...")
    try:
        gan_prep_hf(
            config=config,
            gen_model=generator_model,
            disc_model=discriminator_model,
            gen_tokenizer=generator_tokenizer,
            disc_tokenizer=discriminator_tokenizer,
            tokenized_gen_dataset=tokenized_gen_datasets['train'], # Pass the train split Dataset
            tokenized_disc_dataset=tokenized_disc_datasets['train'] # Pass the train split Dataset
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
            gan_loop_iteration_hf(config) # Call the new function
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
