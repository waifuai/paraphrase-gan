import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import pytest
import tensorflow as tf
from trax import layers as tl
from trax.data import data_streams as streams, text_encoder
from trax.supervised import training


# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
GENERATOR_MODELS = MODELS_ROOT / "generator"
DISCRIMINATOR_MODELS = MODELS_ROOT / "discriminator"

# --- Constants ---
EOS = text_encoder.EOS


# --- Helper Functions ---

def setup_logger(log_dir: Path, log_file: str = "run.log") -> logging.Logger:
    """Sets up a logger with console and file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / log_file
    logger = logging.getLogger("gan_paraphrase")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Console shows INFO and above
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.DEBUG)  # File logs everything
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger(LOGS_DIR)  # Initialize logger


def run_shell_command(
    command: str,
    cwd: str = None,
    capture_output: bool = False,
    shell: bool = False
) -> subprocess.CompletedProcess:
    """
    Runs a shell command with optional working directory and output capture.  Provides more consistent error handling.
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
            shell=shell,  # Use shell=True only for complex shell commands
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr}")
        raise  # Re-raise the exception to stop execution


def ensure_directory(path: Path):
    """Creates a directory if it doesn't exist, handling race conditions."""
    path.mkdir(parents=True, exist_ok=True)


# --- Data Preparation ---
class DataPreparer:
    """Prepares data by reading, cleaning, normalizing, and writing TSV files."""

    def __init__(self, input_dir: Path, output_dir: Path):
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        self.input_dir = input_dir
        self.output_dir = output_dir

    def prepare(self):
        """Prepares the data: validates inputs, cleans output, processes files."""
        self._validate_inputs()
        self._clean_output()
        self._process_files()

    def _validate_inputs(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} not found")

    def _clean_output(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def _process_files(self):
        for file in self.input_dir.glob("*.tsv"):
            self._process_file(file)

    def _process_file(self, file_path: Path):
        output_file = self.output_dir / file_path.name

        with open(file_path, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                cleaned = self._clean_line(line)
                if cleaned:
                    outfile.write(f"{cleaned}\n")
        logger.info(f"Processed {file_path.name} -> {output_file.name}")

    @staticmethod
    def _clean_line(line: str) -> str:
        """Cleans and normalizes a single line of text."""
        line = line.strip().lower()
        if not line or len(line.split("\t")) < 2:
            return ""  # Skip empty or invalid lines
        return line


# --- Trax Problem Definitions ---
class PhraseDiscriminatorProblem:
    """Trax problem: Discriminate human vs. machine generated phrases."""

    def __init__(self, data_dir: str, filename: str = "paraphrases_generated.tsv"):
        self.data_dir = data_dir
        self.filename = filename

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [
            {"split": training.Split.TRAIN, "shards": 1},
            {"split": training.Split.EVAL, "shards": 1},
        ]

    @property
    def approx_vocab_size(self):
        return 2**14

    @property
    def num_classes(self):
        return 2  # human vs machine

    def class_labels(self, data_dir):
        del data_dir  # Unused
        return ["not_human_phrase", "human_phrase"]

    def example_generator(self, filename: str):
        """Generates examples from a TSV file."""
        for line in tf.io.gfile.GFile(filename, "rb"):
            line = text_encoder.to_unicode_utf8(line.strip())
            phrases = line.split("\t")
            if len(phrases) < 2:  # Ensure at least two phrases
                continue

            yield {"inputs": phrases[0], "label": 1}  # First phrase is human

            for n_phrase in range(1, len(phrases)):
                yield {"inputs": phrases[n_phrase], "label": 0}

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generates samples for the problem."""
        filename = os.path.join(self.data_dir, self.filename)
        for example in self.example_generator(filename):
            yield example


class PhraseDiscriminatorProblemCharacters(PhraseDiscriminatorProblem):
    """Character-level PhraseDiscriminatorProblem."""

    @property
    def vocab_type(self):
        return text_encoder.VocabType.CHARACTER


class PhraseGeneratorProblem:
    """Trax problem: Generate a paraphrase for a given phrase."""

    def __init__(self, data_dir:str, filename: str ="paraphrases_selected.tsv"):
        self.EOS = EOS
        self.data_dir = data_dir
        self.filename = filename

    @property
    def approx_vocab_size(self):
        return 2**16  # ~64k

    def dataset_streams(self, data_dir, dataset_split):
        """Data streams for training and evaluation."""
        split = (
            training.Split.TRAIN
            if dataset_split == training.Split.TRAIN
            else training.Split.EVAL
        )
        return streams.TextLineStream(
            os.path.join(self.data_dir, self.filename),
            split=split,
            shuffle=dataset_split == training.Split.TRAIN,
        )

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generates input-target pairs from the dataset."""
        filename = os.path.join(self.data_dir, self.filename)
        with open(filename, "r", encoding="utf-8") as rawfp:
            for curr_line in rawfp:
                curr_line = curr_line.strip().split("\t")
                if len(curr_line) < 2:  # Ensure at least two phrases
                    continue

                for p1, p2 in itertools.permutations(curr_line, 2):
                    yield {"inputs": p1, "targets": p2}


# --- Trax Model Definitions ---
def transformer_phrase_generator(
    d_model=128,
    d_ff=512,
    n_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    attention_dropout=0.6,
    dropout=0.6,
    learning_rate=0.05,
):
    """Creates a Transformer model configuration for phrase generation."""

    return trax.models.Transformer.HParams(
      n_encoder_layers=num_encoder_layers,
      n_decoder_layers=num_decoder_layers,
      d_model=d_model,
      d_ff=d_ff,
      n_heads=n_heads,
      dropout=dropout,
      attention_dropout=attention_dropout,
      learning_rate=learning_rate
    )


def transformer_phrase_generator_range(rhp):
    """Defines hyperparameter ranges for tuning."""
    rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
    rhp.set_int("num_encoder_layers", 2, 8)
    rhp.set_int("num_decoder_layers", 2, 8)
    rhp.set_discrete("d_model", [128, 256, 512])
    rhp.set_float("attention_dropout", 0.4, 0.7)
    rhp.set_discrete("n_heads", [2, 4, 8, 16, 32, 64, 128])
    rhp.set_discrete("d_ff", [512, 1024])


def transformer_phrase_discriminator():
    """Creates a default Transformer model configuration for phrase discrimination."""
    # Placeholder: You can customize this as needed
    return trax.models.Transformer.HParams()


# --- Training and Decoding ---
def train_model(
    problem, model_hparams, data_dir, output_dir, train_steps=1000, eval_steps=100
):
    """Trains a Trax model."""
    ensure_directory(output_dir)

    train_task = training.TrainTask(
        labeled_data=problem.train_stream(data_dir),
        loss_layer=tl.CategoryCrossEntropy(),  # Assuming classification
        optimizer=trax.optimizers.Adam(model_hparams.learning_rate),
        n_steps_per_checkpoint=eval_steps,
    )

    eval_task = training.EvalTask(
        labeled_data=problem.eval_stream(data_dir),
        metrics=[tl.CategoryCrossEntropy(), tl.Accuracy()],  # For classification
    )

    trainer = training.Trainer(
        model=trax.models.Transformer(hparams=model_hparams),
        train_task=train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir,
    )

    trainer.train(n_steps=train_steps)


def decode_model(model_name, problem_name, data_dir, output_dir, model_dir, decode_steps=100):
    """Decodes (infers) using a trained Trax model."""
    ensure_directory(output_dir)

    command = [
        "python",
        "-m",
        "trax.supervised.decode",
        "--output_dir",
        str(output_dir),
        "--model",
        model_name,
        "--problem",
        problem_name,
        "--data_dir",
        str(data_dir),
        "--decode_steps",
        str(decode_steps),
        "--decode_in_memory=True",
    ]
    run_shell_command(command)


# --- Main GAN Workflow Scripts ---

def gan_prep(max_retries=3, retry_delay=5):
    """Prepares the GAN by training initial generator and discriminator models."""
    logger.info("Starting GAN Preparation")

    def train_with_retry(model_type, train_func, *args):
        for attempt in range(1, max_retries + 1):
            try:
                train_func(*args)
                logger.info(f"Training succeeded for {model_type}")
                return  # Exit on success
            except Exception as e:
                logger.warning(
                    f"Training failed for {model_type} (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        logger.error(
            f"Failed to train {model_type} after {max_retries} attempts."
        )
        sys.exit(1)  # Exit after all retries fail.

    # Train Generator
    train_with_retry(
        "generator",
        train_model,
        PhraseGeneratorProblem(PROCESSED_DATA_DIR / "generator"),
        transformer_phrase_generator(),
        PROCESSED_DATA_DIR / "generator",
        GENERATOR_MODELS / "initial",
    )

    # Train Discriminator
    train_with_retry(
        "discriminator",
        train_model,
        PhraseDiscriminatorProblem(PROCESSED_DATA_DIR / "discriminator"),
        transformer_phrase_discriminator(),
        PROCESSED_DATA_DIR / "discriminator",
        DISCRIMINATOR_MODELS / "initial",
    )

    logger.info("GAN Preparation Complete")


def gan_loop_iteration():
    """Runs a single iteration of the GAN training loop."""
    logger.info("Starting GAN Loop Iteration")

    # --- 1. Generate Phrases ---
    generator_data_dir = PROCESSED_DATA_DIR / "generator"
    generator_output_dir = PROCESSED_DATA_DIR / "generator" / "generated"
    ensure_directory(generator_output_dir)

    decode_model(
        "transformer_phrase_generator",
        "phrase_generator_problem",
        generator_data_dir,
        generator_output_dir,
        GENERATOR_MODELS / "latest",  # Use the latest trained generator
    )

    # --- 2. Discriminate Phrases ---
    discriminator_data_dir = generator_output_dir
    discriminator_output_dir = PROCESSED_DATA_DIR / "discriminator" / "classified"
    ensure_directory(discriminator_output_dir)

    decode_model(
        "transformer_phrase_discriminator",
        "phrase_discriminator_problem",
        discriminator_data_dir,
        discriminator_output_dir,
        DISCRIMINATOR_MODELS / "initial", #Keep using the initial discriminator
        decode_steps=1
    )

    # Post-process discriminator output
    postprocess_discriminator_output(
        discriminator_output_dir / "decode_out.txt", #Trax names the output file like this
        discriminator_data_dir / "paraphrases_generated.tsv",
        discriminator_output_dir / "paraphrases_selected.tsv",
    )

    # --- 3. Update Generator Training Data ---
    # Combine original and newly accepted paraphrases
    combine_data(
        generator_data_dir / "paraphrases_selected.tsv",
        discriminator_output_dir / "paraphrases_selected.tsv",
        generator_data_dir / "paraphrases_combined.tsv"
    )

    # --- 4. Train Generator ---

    #Now we retrain on the accepted data + original seed data
    train_model(
        PhraseGeneratorProblem(generator_data_dir, filename="paraphrases_combined.tsv"),
        transformer_phrase_generator(),
        generator_data_dir,
        GENERATOR_MODELS / "latest",  # Overwrite the latest model
    )

    logger.info("GAN Loop Iteration Complete")


def postprocess_discriminator_output(
    labels_file: Path, generated_file: Path, output_file: Path
):
    """Filters generated paraphrases based on discriminator labels."""
    with open(labels_file, "r", encoding="utf-8") as lf, open(
        generated_file, "r", encoding="utf-8"
    ) as gf, open(output_file, "w", encoding="utf-8") as of:
        for label, line in zip(lf, gf):
            if label.strip() == "human_phrase":  # Assuming label output format
                of.write(line)


def combine_data(file1: Path, file2: Path, output_file: Path):
    """Combines two TSV files, ensuring no duplicate lines."""

    combined_lines = set()

    # Read and add lines from both files
    for filepath in [file1, file2]:
        if filepath.exists():  # Check if the file exists
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line: #skip empty lines
                        combined_lines.add(cleaned_line)
        else:
            logger.warning(f"File does not exist: {filepath}")


    # Write unique lines to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in sorted(combined_lines):  # Sorted for consistency
            outfile.write(line + "\n")



# --- Mock Data Generation ---

def generate_mock_paraphrases(output_dir: Path, n_repeat_lines: int = 5000):
    """Generates mock paraphrase data for testing."""
    ensure_directory(output_dir)

    with open(output_dir / "paraphrases_selected.tsv", "w", encoding="utf-8") as f:
        for _ in range(n_repeat_lines):
            f.write("a human phrase\ta human paraphrase\thuman paraphrase\n")
            f.write("another human phrase\tanother human paraphrase\n")

    with open(output_dir / "paraphrases_generated.tsv", "w", encoding="utf-8") as f:
        for _ in range(n_repeat_lines):
            f.write("a human phrase\ta machine phrase\tmachine phrase\n")
            f.write("another human phrase\tanother machine phrase\n")



# --- Main Script Execution ---
def main():
    """Main function to execute the GAN workflow."""
    logger.info("Starting GAN Paraphrase Generation")

    # Create directory structure
    for dir_path in [
        LOGS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        GENERATOR_MODELS,
        DISCRIMINATOR_MODELS,
    ]:
        ensure_directory(dir_path)

    # Generate initial mock data (if it doesn't exist)
    if not (RAW_DATA_DIR / "paraphrases_selected.tsv").exists():
        generate_mock_paraphrases(RAW_DATA_DIR)
    if not (RAW_DATA_DIR / "paraphrases_generated.tsv").exists():
        generate_mock_paraphrases(RAW_DATA_DIR)


    # Prepare initial data
    generator_preparer = DataPreparer(RAW_DATA_DIR, PROCESSED_DATA_DIR / "generator")
    generator_preparer.prepare()

    discriminator_preparer = DataPreparer(RAW_DATA_DIR, PROCESSED_DATA_DIR / "discriminator")
    discriminator_preparer.prepare()

    gan_prep()  # Prepare initial models


    while True:  # Continuous GAN loop
        gan_loop_iteration()


if __name__ == "__main__":
    main()
