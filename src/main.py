import os
import shutil
import subprocess
import sys
import logging
import time
import itertools
from pathlib import Path
from typing import Dict, Any, Set

import trax
from trax import layers as tl
from trax.data import data_streams as streams, text_encoder
from trax.supervised import training

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
        "generator_processed_data": PROJECT_ROOT / "data" / "processed" / "generator",
        "discriminator_processed_data": PROJECT_ROOT / "data" / "processed" / "discriminator",
        "generator_generated_output": PROJECT_ROOT / "data" / "processed" / "generator" / "generated",
        "discriminator_classified_output": PROJECT_ROOT / "data" / "processed" / "discriminator" / "classified",
    },
    "filenames": {
        "log_file": "run.log",
        "generator_selected": "paraphrases_selected.tsv",
        "generator_generated": "paraphrases_generated.tsv", # Used as input for discriminator problem
        "generator_combined": "paraphrases_combined.tsv",
        "discriminator_selected": "paraphrases_selected.tsv", # Output of postprocessing
        "decode_output": "decode_out.txt", # Default Trax decode output filename
    },
    "model_names": {
        "generator": "transformer_phrase_generator",
        "discriminator": "transformer_phrase_discriminator",
    },
    "problem_names": {
        "generator": "PhraseGeneratorProblem",
        "discriminator": "PhraseDiscriminatorProblem",
        "discriminator_chars": "PhraseDiscriminatorProblemCharacters",
    },
    "generator_hparams": {
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "attention_dropout": 0.6,
        "dropout": 0.6,
        "learning_rate": 0.05,
        "vocab_size": 2**16, # approx_vocab_size from PhraseGeneratorProblem
    },
    "discriminator_hparams": {
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2, # Discriminator might not need a decoder, adjust if needed
        "attention_dropout": 0.1,
        "dropout": 0.1,
        "learning_rate": 0.01,
        "vocab_size": 2**14, # approx_vocab_size from PhraseDiscriminatorProblem
        "num_classes": 2,
    },
    "training": {
        "train_steps": 1000,
        "eval_steps": 100,
        "decode_steps": 100,
        "gan_prep_max_retries": 3,
        "gan_prep_retry_delay": 5,
        "mock_data_lines": 5000,
        "discriminator_decode_steps": 1, # Specific decode steps for discriminator classification
    },
    "constants": {
        "EOS": text_encoder.EOS,
    },
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


# --- Trax Problem Definitions ---
# Note: Trax problems are often registered globally. Defining them as classes
# here helps organization, but they might need registration for some Trax tools.
# The decode script uses problem names, so these class names should match
# CONFIG['problem_names'].

class PhraseDiscriminatorProblem(training.Problem):
    """Trax problem: Discriminate human vs. machine generated phrases."""

    def __init__(self, data_dir: str, filename: str, vocab_size: int):
        self.data_dir = data_dir
        self.filename = filename
        self._vocab_size = vocab_size
        logger.info(f"PhraseDiscriminatorProblem initialized: data_dir={data_dir}, filename={filename}")

    @property
    def is_character_level(self):
        return False # Assuming token level unless specified otherwise

    @property
    def input_vocab_size(self):
        return self._vocab_size

    @property
    def target_vocab_size(self):
         # For classification problems, target_vocab_size is num_classes
        return CONFIG['discriminator_hparams']['num_classes']

    @property
    def num_classes(self):
        return CONFIG['discriminator_hparams']['num_classes']

    def class_labels(self, data_dir=None):
        del data_dir # Unused
        return ["not_human_phrase", "human_phrase"]

    def examples_stream(self, data_dir=None, which_set='train'):
        """Generates examples from a TSV file."""
        # Note: Trax typically uses dataset_filename pattern or specific splits.
        # This implementation reads the whole file for simplicity here.
        # For proper train/eval split, more sophisticated handling is needed.
        filepath = Path(self.data_dir) / self.filename
        logger.debug(f"Generating examples for Discriminator from: {filepath}")
        if not filepath.exists():
            logger.error(f"Discriminator data file not found: {filepath}")
            raise FileNotFoundError(f"Discriminator data file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    phrases = line.split("\t")
                    if len(phrases) < 2:
                        logger.warning(f"Skipping line with < 2 phrases: '{line[:50]}...'")
                        continue

                    # Yield the "human" phrase (assuming first column is original/human)
                    # The label '1' corresponds to 'human_phrase' in class_labels
                    yield (phrases[0], 1)

                    # Yield the "machine" phrases (subsequent columns)
                    # The label '0' corresponds to 'not_human_phrase'
                    for n_phrase in range(1, len(phrases)):
                        yield (phrases[n_phrase], 0)
        except Exception as e:
            logger.error(f"Error reading discriminator data file {filepath}: {e}")
            raise


class PhraseDiscriminatorProblemCharacters(PhraseDiscriminatorProblem):
    """Character-level PhraseDiscriminatorProblem."""
    @property
    def is_character_level(self):
        return True


class PhraseGeneratorProblem(training.Problem):
    """Trax problem: Generate a paraphrase for a given phrase."""

    def __init__(self, data_dir: str, filename: str, vocab_size: int):
        self.data_dir = data_dir
        self.filename = filename
        self._vocab_size = vocab_size
        self.EOS = CONFIG['constants']['EOS']
        logger.info(f"PhraseGeneratorProblem initialized: data_dir={data_dir}, filename={filename}")

    @property
    def is_character_level(self):
        return False # Assuming token level

    @property
    def input_vocab_size(self):
        return self._vocab_size

    @property
    def target_vocab_size(self):
        return self._vocab_size # Generator target is also text

    def examples_stream(self, data_dir=None, which_set='train'):
        """Generates input-target pairs from the dataset."""
        # Again, simplified to read the whole file. Needs proper split handling.
        filepath = Path(self.data_dir) / self.filename
        logger.debug(f"Generating examples for Generator from: {filepath}")
        if not filepath.exists():
            logger.error(f"Generator data file not found: {filepath}")
            raise FileNotFoundError(f"Generator data file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as rawfp:
                for curr_line in rawfp:
                    curr_line = curr_line.strip().split("\t")
                    if len(curr_line) < 2:
                        logger.warning(f"Skipping line with < 2 phrases: '{curr_line[0][:50] if curr_line else ''}...'")
                        continue

                    # Create all permutations of pairs as input/target
                    for p1, p2 in itertools.permutations(curr_line, 2):
                        yield (p1, p2) # Yield tuple (input, target)
        except Exception as e:
            logger.error(f"Error reading generator data file {filepath}: {e}")
            raise


# --- Trax Model Definitions ---
# These functions now return the model *instance* with HParams applied.

def transformer_phrase_generator(hparams: Dict[str, Any], mode: str = 'train'):
    """Creates a Transformer model for phrase generation."""
    logger.debug(f"Creating generator model with hparams: {hparams}")
    # Convert dict to HParams object if necessary, or pass dict directly if supported
    # Assuming trax.models.Transformer can accept a dict or requires HParams
    try:
        hparams_obj = trax.models.Transformer.HParams(**hparams)
    except TypeError:
         # Fallback if HParams doesn't accept dict directly
         hparams_obj = trax.models.Transformer.HParams()
         for k, v in hparams.items():
             if hasattr(hparams_obj, k):
                 setattr(hparams_obj, k, v)
             else:
                 logger.warning(f"Ignoring unknown hparam '{k}' for Transformer")

    return trax.models.Transformer(
        d_model=hparams_obj.d_model,
        d_ff=hparams_obj.d_ff,
        n_encoder_layers=hparams_obj.n_encoder_layers,
        n_decoder_layers=hparams_obj.n_decoder_layers,
        n_heads=hparams_obj.n_heads,
        dropout=hparams_obj.dropout,
        attention_dropout=hparams_obj.attention_dropout,
        vocab_size=hparams_obj.vocab_size,
        mode=mode,
        # Add other necessary parameters if needed based on Trax version
    )


def transformer_phrase_discriminator(hparams: Dict[str, Any], mode: str = 'train'):
    """Creates a Transformer model for phrase discrimination (classification)."""
    logger.debug(f"Creating discriminator model with hparams: {hparams}")
    # Convert dict to HParams object
    try:
        hparams_obj = trax.models.Transformer.HParams(**hparams)
    except TypeError:
         hparams_obj = trax.models.Transformer.HParams()
         for k, v in hparams.items():
             if hasattr(hparams_obj, k):
                 setattr(hparams_obj, k, v)
             else:
                 logger.warning(f"Ignoring unknown hparam '{k}' for Transformer")

    # For classification, we typically use the encoder part and add a classification head.
    # Trax might have a specific classification model or require manual head addition.
    # This example uses TransformerLM as a base, which might need adjustment.
    # A simpler sequence classification model might be more appropriate.
    # Let's assume a simple approach: Transformer encoder + Dense layer + LogSoftmax
    return tl.Serial(
        trax.models.TransformerEncoder(
            vocab_size=hparams_obj.vocab_size,
            d_model=hparams_obj.d_model,
            d_ff=hparams_obj.d_ff,
            n_layers=hparams_obj.n_encoder_layers, # Use encoder layers
            n_heads=hparams_obj.n_heads,
            dropout=hparams_obj.dropout,
            attention_dropout=hparams_obj.attention_dropout,
            mode=mode,
        ),
        tl.Select([0]), # Select the output of the encoder
        tl.Mean(axis=1), # Average pooling over sequence length
        tl.Dense(hparams_obj.num_classes),
        tl.LogSoftmax()
    )


# --- Training and Decoding ---
def train_model(
    problem: training.Problem,
    model_fn: callable, # Function to create the model (e.g., transformer_phrase_generator)
    model_hparams: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
):
    """Trains a Trax model."""
    ensure_directory(output_dir)
    logger.info(f"Starting training for {problem.__class__.__name__}...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model HParams: {model_hparams}")
    logger.info(f"Training steps: {config['training']['train_steps']}, Eval steps: {config['training']['eval_steps']}")

    # Determine loss layer based on problem type
    if isinstance(problem, PhraseGeneratorProblem):
        loss_layer = tl.CrossEntropyLoss()
        metrics = [tl.CrossEntropyLoss(), tl.Accuracy()] # Token accuracy
    elif isinstance(problem, PhraseDiscriminatorProblem):
        loss_layer = tl.CategoryCrossEntropy()
        metrics = [tl.CategoryCrossEntropy(), tl.Accuracy()] # Classification accuracy
    else:
        logger.error(f"Unknown problem type for loss determination: {type(problem)}")
        raise ValueError("Unknown problem type")

    lr_schedule = trax.lr.multifactor(
        constant=model_hparams.get('learning_rate', 0.01), # Default LR if not in hparams
        # Add other schedule params if needed (warmup, decay, etc.)
    )

    train_task = training.TrainTask(
        labeled_data=problem.train_stream(1), # Batch size 1 for stream
        loss_layer=loss_layer,
        optimizer=trax.optimizers.Adam(learning_rate=model_hparams.get('learning_rate', 0.01)), # Get LR from hparams
        # lr_schedule=lr_schedule, # Use LR schedule if defined
        n_steps_per_checkpoint=config['training']['eval_steps'],
    )

    eval_task = training.EvalTask(
        labeled_data=problem.eval_stream(1), # Batch size 1 for stream
        metrics=metrics,
        n_eval_batches=10,  # Number of batches to use for evaluation
    )

    loop = training.Loop(
        model_fn(model_hparams, mode='train'),
        tasks=[train_task],
        eval_tasks=[eval_task],
        output_dir=output_dir,
        # checkpoint_at=lambda step: step % config['training']['eval_steps'] == 0, # Redundant with n_steps_per_checkpoint
    )

    try:
        loop.run(n_steps=config['training']['train_steps'])
        logger.info(f"Training complete for {problem.__class__.__name__}.")
    except Exception as e:
        logger.error(f"Training failed for {problem.__class__.__name__}: {e}")
        raise


def decode_model(
    model_name: str,
    problem_name: str, # Use the registered problem name or class name
    data_dir: Path, # Directory containing the input file for decoding
    input_filename: str, # Specific input file within data_dir
    output_dir: Path,
    model_dir: Path, # Directory where the trained model checkpoint is
    config: Dict[str, Any],
    decode_steps: int,
):
    """Decodes (infers) using a trained Trax model via subprocess."""
    ensure_directory(output_dir)
    logger.info(f"Starting decoding for model {model_name}...")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Input data directory: {data_dir}")
    logger.info(f"Input filename: {input_filename}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Decode steps: {decode_steps}")

    # Construct the command carefully
    command = [
        sys.executable, # Use the current python interpreter
        "-m", "trax.supervised.decode",
        f"--model={model_name}",
        f"--problem={problem_name}",
        f"--data_dir={str(data_dir)}",
        f"--output_dir={str(output_dir)}",
        f"--model_dir={str(model_dir)}", # Specify the directory containing checkpoints
        f"--input_file={input_filename}", # Specify the input file relative to data_dir
        f"--output_file={config['filenames']['decode_output']}", # Specify the output filename
        f"--decode_steps={decode_steps}",
        "--decode_from_file=True", # Decode from the specified input file
        "--decode_to_file=True", # Decode to the specified output file
        # "--decode_in_memory=True", # Avoid if decoding from/to file
        # Add other necessary flags like --batch_size if needed
    ]

    try:
        run_shell_command(command)
        logger.info(f"Decoding complete for {model_name}. Output in {output_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Decoding failed for {model_name}: {e}")
        # Check if output file exists despite error
        output_file_path = output_dir / config['filenames']['decode_output']
        if not output_file_path.exists():
             logger.error("Decode output file was not created.")
        else:
             logger.warning("Decode output file exists, but the process reported an error.")
        raise # Re-raise the exception


# --- Main GAN Workflow Scripts ---

def gan_prep(config: Dict[str, Any]):
    """Prepares the GAN by training initial generator and discriminator models."""
    logger.info("--- Starting GAN Preparation ---")
    max_retries = config['training']['gan_prep_max_retries']
    retry_delay = config['training']['gan_prep_retry_delay']

    def train_with_retry(model_type: str, problem_instance: training.Problem, model_fn: callable, hparams: Dict[str, Any], output_dir: Path):
        """Inner function to handle training with retries."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{max_retries} to train {model_type}...")
                train_model(
                    problem_instance,
                    model_fn,
                    hparams,
                    output_dir,
                    config,
                )
                logger.info(f"Training succeeded for {model_type} on attempt {attempt}.")
                return  # Exit on success
            except Exception as e:
                logger.warning(
                    f"Training failed for {model_type} (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to train {model_type} after {max_retries} attempts."
                    )
                    raise  # Re-raise the final exception

    # --- Train Initial Generator ---
    logger.info("Training initial generator...")
    gen_problem = PhraseGeneratorProblem(
        data_dir=config['paths']['generator_processed_data'],
        filename=config['filenames']['generator_selected'],
        vocab_size=config['generator_hparams']['vocab_size']
    )
    try:
        train_with_retry(
            "generator",
            gen_problem,
            transformer_phrase_generator,
            config['generator_hparams'],
            config['paths']['generator_initial_model'],
        )
        # Copy initial model to latest for the first loop iteration
        initial_gen_path = config['paths']['generator_initial_model']
        latest_gen_path = config['paths']['generator_latest_model']
        if latest_gen_path.exists():
            logger.warning(f"Removing existing latest generator model at: {latest_gen_path}")
            shutil.rmtree(latest_gen_path)
        shutil.copytree(initial_gen_path, latest_gen_path)
        logger.info(f"Copied initial generator model to {latest_gen_path}")

    except Exception as e:
        logger.error(f"GAN Prep failed during initial generator training: {e}")
        sys.exit(1) # Exit if initial training fails

    # --- Train Initial Discriminator ---
    logger.info("Training initial discriminator...")
    disc_problem = PhraseDiscriminatorProblem(
        data_dir=config['paths']['discriminator_processed_data'],
        filename=config['filenames']['generator_generated'], # Train on distinguishing initial mock data
        vocab_size=config['discriminator_hparams']['vocab_size']
    )
    try:
        train_with_retry(
            "discriminator",
            disc_problem,
            transformer_phrase_discriminator,
            config['discriminator_hparams'],
            config['paths']['discriminator_initial_model'],
        )
    except Exception as e:
        logger.error(f"GAN Prep failed during initial discriminator training: {e}")
        sys.exit(1) # Exit if initial training fails

    logger.info("--- GAN Preparation Complete ---")


def gan_loop_iteration(config: Dict[str, Any]):
    """Runs a single iteration of the GAN training loop."""
    logger.info("--- Starting GAN Loop Iteration ---")

    # Define paths for this iteration
    gen_data_dir = config['paths']['generator_processed_data']
    gen_output_dir = config['paths']['generator_generated_output'] # Where generated phrases go
    disc_classified_dir = config['paths']['discriminator_classified_output'] # Where classification results go
    latest_gen_model_dir = config['paths']['generator_latest_model']
    initial_disc_model_dir = config['paths']['discriminator_initial_model']

    # --- 1. Generate Phrases using the *latest* generator ---
    logger.info("Step 1: Generating Phrases...")
    ensure_directory(gen_output_dir)
    # Input for generation is the 'selected' file (human phrases to paraphrase)
    generation_input_file = config['filenames']['generator_selected']
    try:
        decode_model(
            model_name=config['model_names']['generator'],
            problem_name=config['problem_names']['generator'],
            data_dir=gen_data_dir, # Directory containing the input file
            input_filename=generation_input_file,
            output_dir=gen_output_dir, # Output *directory* for decode_out.txt
            model_dir=latest_gen_model_dir,
            config=config,
            decode_steps=config['training']['decode_steps'],
        )
        # Rename the output file for clarity and consistency
        generated_phrases_file = gen_output_dir / config['filenames']['generator_generated']
        default_decode_output = gen_output_dir / config['filenames']['decode_output']
        if default_decode_output.exists():
             default_decode_output.rename(generated_phrases_file)
             logger.info(f"Renamed generator output to {generated_phrases_file}")
        else:
             logger.error(f"Generator decode output file not found: {default_decode_output}")
             raise FileNotFoundError("Generator decode output missing")

    except Exception as e:
        logger.error(f"Failed Step 1 (Generate Phrases): {e}")
        raise # Stop iteration if generation fails

    # --- 2. Discriminate Phrases using the *initial* discriminator ---
    logger.info("Step 2: Discriminating Phrases...")
    ensure_directory(disc_classified_dir)
    # Input for discrimination is the file *just generated* by the generator
    discrimination_input_file = config['filenames']['generator_generated']
    try:
        decode_model(
            model_name=config['model_names']['discriminator'],
            problem_name=config['problem_names']['discriminator'],
            data_dir=gen_output_dir, # Directory containing the generated phrases
            input_filename=discrimination_input_file,
            output_dir=disc_classified_dir, # Output *directory* for classification labels
            model_dir=initial_disc_model_dir,
            config=config,
            decode_steps=config['training']['discriminator_decode_steps'], # Usually just 1 step for classification
        )
        # The output file (labels) is named decode_out.txt by default
        labels_file = disc_classified_dir / config['filenames']['decode_output']
        if not labels_file.exists():
             logger.error(f"Discriminator decode output file not found: {labels_file}")
             raise FileNotFoundError("Discriminator decode output missing")

    except Exception as e:
        logger.error(f"Failed Step 2 (Discriminate Phrases): {e}")
        raise # Stop iteration if discrimination fails

    # --- 3. Post-process Discriminator Output ---
    logger.info("Step 3: Post-processing Discriminator Output...")
    # Input labels: labels_file from Step 2
    # Input generated phrases: generated_phrases_file from Step 1
    # Output selected phrases: discriminator_selected_file
    discriminator_selected_file = disc_classified_dir / config['filenames']['discriminator_selected']
    try:
        postprocess_discriminator_output(
            labels_file=labels_file,
            generated_file=generated_phrases_file, # File containing the phrases that were classified
            output_file=discriminator_selected_file,
            human_label="human_phrase", # Label corresponding to '1' in the problem
        )
    except Exception as e:
        logger.error(f"Failed Step 3 (Post-process Discriminator Output): {e}")
        raise

    # --- 4. Update Generator Training Data ---
    logger.info("Step 4: Updating Generator Training Data...")
    # Combine original selected data with newly selected (human-classified) generated data
    original_selected_file = gen_data_dir / config['filenames']['generator_selected']
    newly_selected_file = discriminator_selected_file # Output from Step 3
    combined_training_file = gen_data_dir / config['filenames']['generator_combined']
    try:
        combine_data(
            original_selected_file,
            newly_selected_file,
            combined_training_file,
        )
    except Exception as e:
        logger.error(f"Failed Step 4 (Combine Data): {e}")
        raise

    # --- 5. Train Generator (Update Latest) ---
    logger.info("Step 5: Training Generator...")
    # Train on the newly combined dataset
    gen_problem_updated = PhraseGeneratorProblem(
        data_dir=gen_data_dir,
        filename=config['filenames']['generator_combined'], # Use the combined file
        vocab_size=config['generator_hparams']['vocab_size']
    )
    try:
        # Train and overwrite the 'latest' generator model
        train_model(
            gen_problem_updated,
            transformer_phrase_generator,
            config['generator_hparams'],
            latest_gen_model_dir, # Output to 'latest' directory
            config,
        )
    except Exception as e:
        logger.error(f"Failed Step 5 (Train Generator): {e}")
        # Decide whether to stop or continue loop on training failure
        raise # Stop iteration if training fails

    logger.info("--- GAN Loop Iteration Complete ---")


def postprocess_discriminator_output(
    labels_file: Path, generated_file: Path, output_file: Path, human_label: str
):
    """Filters generated paraphrases based on discriminator labels."""
    logger.info(f"Post-processing: Labels='{labels_file}', Generated='{generated_file}', Output='{output_file}'")
    lines_written = 0
    lines_read = 0
    if not labels_file.exists():
        logger.error(f"Labels file not found: {labels_file}")
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if not generated_file.exists():
        logger.error(f"Generated phrases file not found: {generated_file}")
        raise FileNotFoundError(f"Generated phrases file not found: {generated_file}")

    try:
        with open(labels_file, "r", encoding="utf-8") as lf, \
             open(generated_file, "r", encoding="utf-8") as gf, \
             open(output_file, "w", encoding="utf-8") as of:

            # Read generated phrases into a list first, as label file might be shorter
            # if decode_steps was less than the number of lines in generated_file
            generated_lines = gf.readlines()

            for i, label in enumerate(lf):
                lines_read += 1
                label_clean = label.strip()
                if i < len(generated_lines):
                    if label_clean == human_label:
                        of.write(generated_lines[i]) # Write the corresponding original line
                        lines_written += 1
                else:
                    logger.warning(f"Label file '{labels_file.name}' has more lines ({lines_read}) than generated file '{generated_file.name}' ({len(generated_lines)}). Stopping processing.")
                    break # Stop if label file is longer than generated file

        logger.info(f"Post-processing complete. Read {lines_read} labels, selected {lines_written} lines.")
        if lines_read < len(generated_lines):
             logger.warning(f"Label file '{labels_file.name}' ({lines_read} lines) was shorter than generated file '{generated_file.name}' ({len(generated_lines)} lines). Some generated phrases were not classified.")

    except Exception as e:
        logger.error(f"Error during post-processing: {e}")
        raise


def combine_data(file1: Path, file2: Path, output_file: Path):
    """Combines two TSV files, ensuring no duplicate lines."""
    logger.info(f"Combining data: '{file1.name}' + '{file2.name}' -> '{output_file.name}'")
    combined_lines: Set[str] = set()
    files_read = 0
    lines_added = 0

    for filepath in [file1, file2]:
        if filepath.exists():
            files_read += 1
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    initial_count = len(combined_lines)
                    for line in f:
                        cleaned_line = line.strip()
                        if cleaned_line: # Skip empty lines
                            combined_lines.add(cleaned_line)
                    lines_read_file = len(combined_lines) - initial_count
                    logger.debug(f"Read {lines_read_file} unique lines from {filepath.name}")
            except Exception as e:
                 logger.error(f"Error reading file {filepath} during combination: {e}")
                 # Decide if error is fatal or skippable
                 # raise # Uncomment to make it fatal
        else:
            logger.warning(f"Combine data: File does not exist, skipping: {filepath}")

    if not combined_lines:
         logger.warning("No lines found in input files to combine. Output file will be empty.")


    # Write unique lines to the output file
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            # Sort for consistency, although set inherently removes order
            for line in sorted(list(combined_lines)):
                outfile.write(line + "\n")
                lines_added += 1
        logger.info(f"Combine data complete. Read {files_read} files, wrote {lines_added} unique lines to {output_file.name}")
    except Exception as e:
        logger.error(f"Error writing combined data to {output_file}: {e}")
        raise


# --- Mock Data Generation ---

def generate_mock_paraphrases(output_dir: Path, config: Dict[str, Any]):
    """Generates mock paraphrase data based on config."""
    ensure_directory(output_dir)
    n_repeat_lines = config['training']['mock_data_lines']
    selected_file = output_dir / config['filenames']['generator_selected']
    generated_file = output_dir / config['filenames']['generator_generated']
    logger.info(f"Generating mock data ({n_repeat_lines} lines) in {output_dir}...")

    try:
        # Mock "selected" (human-like) data
        with open(selected_file, "w", encoding="utf-8") as f_sel:
            for i in range(n_repeat_lines):
                f_sel.write(f"human phrase {i+1}\thuman paraphrase {i+1}a\thuman paraphrase {i+1}b\n")
                # Add more variety if needed
                if i % 2 == 0:
                     f_sel.write(f"another human phrase {i+1}\tanother good paraphrase {i+1}\n")

        # Mock "generated" (mix of human/machine) data for initial discriminator training
        with open(generated_file, "w", encoding="utf-8") as f_gen:
            for i in range(n_repeat_lines):
                # Line simulating human source + machine outputs
                f_gen.write(f"human phrase {i+1}\tmachine phrase {i+1}x\tmachine phrase {i+1}y\n")
                 # Line simulating another human source + machine outputs
                if i % 3 == 0:
                     f_gen.write(f"another human phrase {i+1}\tanother machine attempt {i+1}\n")

        logger.info(f"Generated mock files: {selected_file.name}, {generated_file.name}")
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        raise


# --- Main Script Execution ---
def main(config: Dict[str, Any]):
    """Main function to execute the GAN workflow."""
    logger.info("========== Starting GAN Paraphrase Generation ==========")

    # Create essential directory structure from config
    paths = config['paths']
    ensure_directory(paths['logs_dir'])
    ensure_directory(paths['raw_data_dir'])
    ensure_directory(paths['processed_data_dir'])
    ensure_directory(paths['generator_models'])
    ensure_directory(paths['discriminator_models'])
    ensure_directory(paths['generator_processed_data'])
    ensure_directory(paths['discriminator_processed_data'])
    # Initial/Latest model dirs will be created by training/copying

    # Generate initial mock data (if raw data files don't exist)
    raw_selected = paths['raw_data_dir'] / config['filenames']['generator_selected']
    raw_generated = paths['raw_data_dir'] / config['filenames']['generator_generated']
    if not raw_selected.exists() or not raw_generated.exists():
        logger.info("Raw data files not found. Generating mock data...")
        try:
            generate_mock_paraphrases(paths['raw_data_dir'], config)
        except Exception as e:
            logger.critical(f"Failed to generate mock data. Cannot proceed. Error: {e}")
            sys.exit(1)
    else:
        logger.info("Raw data files found. Skipping mock data generation.")


    # Prepare initial data (copy/clean raw data to processed directories)
    logger.info("Preparing initial processed data...")
    try:
        generator_preparer = DataPreparer(paths['raw_data_dir'], paths['generator_processed_data'])
        generator_preparer.prepare()

        discriminator_preparer = DataPreparer(paths['raw_data_dir'], paths['discriminator_processed_data'])
        discriminator_preparer.prepare()
    except Exception as e:
        logger.critical(f"Failed during initial data preparation. Cannot proceed. Error: {e}")
        sys.exit(1)

    # Train initial models (GAN Prep)
    try:
        gan_prep(config)
    except Exception as e:
         logger.critical(f"GAN preparation (initial training) failed. Cannot proceed. Error: {e}")
         sys.exit(1)

    # Continuous GAN loop
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"========== Starting GAN Iteration {iteration} ==========")
        try:
            gan_loop_iteration(config)
            logger.info(f"========== Completed GAN Iteration {iteration} ==========")
            # Optional: Add a delay between iterations
            # time.sleep(config['training'].get('loop_delay', 5))
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting GAN loop.")
            break
        except Exception as e:
            logger.error(f"Error during GAN loop iteration {iteration}: {e}")
            logger.error("Attempting to continue to the next iteration...")
            # Optional: Add a longer delay or exit strategy on repeated errors
            time.sleep(10) # Delay before next attempt


if __name__ == "__main__":
    try:
        main(CONFIG)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in main: {e}", exc_info=True)
        sys.exit(1)
