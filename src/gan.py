import sys
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
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
from datasets import Dataset, load_dataset

# Import necessary components from other modules
from .config import CONFIG # Assuming config is needed here, adjust if not
from .utils import logger, ensure_directory
from .model_utils import train_hf_model, load_discriminator_model # Assuming load_discriminator_model is needed here
from .data_utils import postprocess_discriminator_output_hf, combine_data_hf

# --- Main GAN Workflow Scripts ---

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


def gan_loop_iteration_hf(
    config: Dict[str, Any],
    gen_tokenizer: AutoTokenizer, # Pass tokenizer explicitly
    disc_tokenizer: AutoTokenizer, # Pass tokenizer explicitly
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
             return gen_tokenizer(examples["input_phrase"], padding=False, truncation=True, max_length=training_config['generator_max_length']) # No padding needed for generate

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
        generated_phrase_texts = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
        disc_model.to(device)

        # Load the generated phrases file into a dataset
        generated_ds = load_dataset("csv", data_files={"predict": str(discriminator_input_generated_file)}, delimiter="\t")['predict']

        def tokenize_disc_input(examples):
             phrases = examples["phrase"]
             phrases = ["" if p is None else p for p in phrases] # Handle potential None
             return disc_tokenizer(phrases, padding="max_length", truncation=True, max_length=training_config['discriminator_max_length'])

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
        disc_data_collator = DataCollatorWithPadding(tokenizer=disc_tokenizer)
        trainer = Trainer(
            model=disc_model,
            args=predict_args,
            tokenizer=disc_tokenizer,
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
            model_inputs = gen_tokenizer(inputs, max_length=training_config['generator_max_length'], truncation=True)
            with gen_tokenizer.as_target_tokenizer():
                labels = gen_tokenizer(targets, max_length=training_config['generator_max_length'], truncation=True)
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
        gen_data_collator = DataCollatorForSeq2Seq(tokenizer=gen_tokenizer, model=gen_model)

        # Train the generator model (loaded at the start of the loop)
        train_hf_model(
            model=gen_model,
            tokenizer=gen_tokenizer,
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