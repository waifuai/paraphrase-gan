import sys
from typing import Any
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import necessary components from other modules
from .utils import logger

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
    training_args: TrainingArguments,
    data_collator: Any, # DataCollatorForSeq2Seq or DataCollatorWithPadding
    eval_dataset: Dataset = None, # Optional eval dataset
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