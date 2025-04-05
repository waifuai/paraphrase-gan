import shutil
from pathlib import Path
from typing import Dict, Any, Set, List
import numpy as np

# Import necessary components from other modules
from .config import CONFIG
from .utils import logger, ensure_directory

# --- Data Preparation Class ---

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

# --- GAN Loop Data Handling ---

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