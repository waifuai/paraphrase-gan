import logging
from pathlib import Path
from typing import Dict, Any

# Define project root based on this file's location
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
    "training": {
        # Adjusted training params for HF Trainer (epochs are more common than steps)
        "num_train_epochs": 1, # Example: train for 1 epoch initially
        "per_device_train_batch_size": 8, # Example batch size
        "per_device_eval_batch_size": 8, # Example batch size
        "logging_steps": 100, # Log every 100 steps
        "eval_steps": 100, # Evaluate every 100 steps (can be evaluation_strategy='steps')
        "save_steps": 500, # Save checkpoint every 500 steps
        "gan_prep_max_retries": 3,
        "gan_prep_retry_delay": 5,
        "mock_data_lines": 500, # Reduced for faster testing/debugging
        "discriminator_decode_steps": 1, # Specific decode steps for discriminator classification - Not directly applicable to HF predict
        "generator_max_length": 128,
        "discriminator_max_length": 128,
        "generator_update_learning_rate": 2e-5, # Potentially smaller LR for updates
    },
    "logging": {
        "logger_name": "gan_paraphrase",
        "console_level": logging.INFO,
        "file_level": logging.DEBUG,
    }
}