import pytest
import subprocess
import os
import csv
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset

# Import necessary components from src.main
# Note: We might need to adjust imports if functions are nested or rely on global state
from src.main import (
    CONFIG,
    ensure_directory,
    generate_mock_paraphrases,
    load_generator_model,
    load_discriminator_model,
    # Preprocessing functions are defined inside main, test them via map
    postprocess_discriminator_output_hf,
    combine_data_hf,
    generator_tokenizer, # Assuming global tokenizer access for tests
    discriminator_tokenizer # Assuming global tokenizer access for tests
)
from transformers import (
    T5ForConditionalGeneration,
    BertForSequenceClassification,
    AutoTokenizer # Re-import for clarity if needed
)

# --- Test Fixtures (from conftest.py) ---
# clean_test_env fixture is used automatically by pytest

# --- Helper Function for Tests ---
def count_lines(filepath: Path, skip_header=False):
    """Counts lines in a file, optionally skipping the header."""
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    count = len(lines)
    if skip_header and count > 0:
        return count - 1
    return count

# --- Test Cases ---

def test_generate_mock_paraphrases(clean_test_env):
    """Tests the generation of mock data files."""
    mock_data_dir = clean_test_env / "mock_data"
    test_config = {
        "training": {"mock_data_lines": 10}, # Use small number for test
        "filenames": CONFIG['filenames'] # Use actual filenames
    }
    generate_mock_paraphrases(mock_data_dir, test_config)

    gen_file = mock_data_dir / test_config['filenames']['mock_generator_input']
    disc_file = mock_data_dir / test_config['filenames']['mock_discriminator_input']

    assert gen_file.exists()
    assert disc_file.exists()

    # Check generator file header and line count (approximate, depends on pairs generated)
    with open(gen_file, "r", encoding="utf-8") as f:
        gen_reader = csv.reader(f, delimiter="\t")
        header = next(gen_reader)
        assert header == ["input_phrase", "target_phrase"]
    # Expect roughly 4 lines per n_line (2 pairs, 2 directions) + header
    assert count_lines(gen_file) > test_config['training']['mock_data_lines'] * 2

    # Check discriminator file header and line count
    with open(disc_file, "r", encoding="utf-8") as f:
        disc_reader = csv.reader(f, delimiter="\t")
        header = next(disc_reader)
        assert header == ["phrase", "label"]
    # Expect roughly 4 lines per n_line (2 human, 2 machine) + header
    assert count_lines(disc_file) > test_config['training']['mock_data_lines'] * 2


def test_data_loading_and_preprocessing(clean_test_env):
    """Tests loading mock data and applying preprocessing."""
    mock_data_dir = clean_test_env / "mock_data"
    test_config = {
        "training": {
            "mock_data_lines": 5,
            "generator_max_length": 32, # Use smaller max_length for testing
            "discriminator_max_length": 32
        },
        "filenames": CONFIG['filenames']
    }
    generate_mock_paraphrases(mock_data_dir, test_config)

    raw_gen_input_path = mock_data_dir / test_config['filenames']['mock_generator_input']
    raw_disc_input_path = mock_data_dir / test_config['filenames']['mock_discriminator_input']

    # Load datasets
    gen_data_files = {"train": str(raw_gen_input_path)}
    raw_gen_datasets = load_dataset("csv", data_files=gen_data_files, delimiter="\t")
    disc_data_files = {"train": str(raw_disc_input_path)}
    raw_disc_datasets = load_dataset("csv", data_files=disc_data_files, delimiter="\t")

    # Define preprocessing functions locally for testing (mirroring main.py)
    gen_max_len = test_config['training']['generator_max_length']
    disc_max_len = test_config['training']['discriminator_max_length']

    def preprocess_generator(examples):
        inputs = examples["input_phrase"]
        targets = examples["target_phrase"]
        inputs = ["" if i is None else i for i in inputs]
        targets = ["" if t is None else t for t in targets]
        model_inputs = generator_tokenizer(
            inputs,
            text_target=targets,
            max_length=gen_max_len,
            truncation=True
        )
        # The tokenizer automatically creates the 'labels' field when text_target is provided
        return model_inputs

    def preprocess_discriminator(examples):
        phrases = examples["phrase"]
        phrases = ["" if p is None else p for p in phrases]
        tokenized_inputs = discriminator_tokenizer(phrases, truncation=True, max_length=disc_max_len)
        tokenized_inputs["labels"] = examples["label"] # Keep labels
        return tokenized_inputs

    # Apply preprocessing
    tokenized_gen = raw_gen_datasets.map(preprocess_generator, batched=True, remove_columns=raw_gen_datasets["train"].column_names)
    tokenized_disc = raw_disc_datasets.map(preprocess_discriminator, batched=True, remove_columns=["phrase"]) # Keep label

    # Basic checks
    assert "input_ids" in tokenized_gen["train"].features
    assert "labels" in tokenized_gen["train"].features
    assert "input_ids" in tokenized_disc["train"].features
    assert "labels" in tokenized_disc["train"].features # Check label is present
    assert "phrase" not in tokenized_disc["train"].features # Check phrase is removed

    # Check tokenization output length (example)
    assert len(tokenized_gen["train"][0]["input_ids"]) <= gen_max_len
    assert len(tokenized_disc["train"][0]["input_ids"]) <= disc_max_len


def test_model_loading():
    """Tests loading the specified HF models."""
    gen_model = load_generator_model(CONFIG['model_identifiers']['generator'])
    disc_model = load_discriminator_model(CONFIG['model_identifiers']['discriminator'])

    assert isinstance(gen_model, T5ForConditionalGeneration)
    assert isinstance(disc_model, BertForSequenceClassification)
    assert disc_model.config.num_labels == 2 # Check if num_labels is set


def test_postprocess_discriminator_output_hf(clean_test_env):
    """Tests filtering and formatting of generated phrases based on predictions."""
    output_file = clean_test_env / "selected_output.tsv"
    predictions = np.array([1, 0, 1, 0, 1]) # Human, Machine, Human, Machine, Human
    unique_inputs = [
        "paraphrase: input 1",
        "paraphrase: input 2",
        "paraphrase: input 3",
        "paraphrase: input 4",
        "paraphrase: input 5"
    ]
    generated_phrases = [
        "generated 1",
        "generated 2",
        "generated 3",
        "generated 4",
        "generated 5"
    ]

    postprocess_discriminator_output_hf(
        predictions=predictions,
        unique_input_phrases=unique_inputs,
        generated_phrases=generated_phrases,
        output_file=output_file,
        human_label_index=1
    )

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split('\n')
    assert len(lines) == 4 # Header + 3 selected lines
    assert lines[0] == "input_phrase\ttarget_phrase"
    assert lines[1] == "paraphrase: input 1\tgenerated 1"
    assert lines[2] == "paraphrase: input 3\tgenerated 3"
    assert lines[3] == "paraphrase: input 5\tgenerated 5"


def test_combine_data_hf(clean_test_env):
    """Tests combining original and selected generated data."""
    original_file = clean_test_env / "original.tsv"
    selected_file = clean_test_env / "selected.tsv"
    output_file = clean_test_env / "combined.tsv"

    # Create dummy original data
    with open(original_file, "w", encoding="utf-8") as f:
        f.write("input_phrase\ttarget_phrase\n")
        f.write("paraphrase: original 1\toriginal 1a\n")
        f.write("paraphrase: original 2\toriginal 2a\n")

    # Create dummy selected generated data
    with open(selected_file, "w", encoding="utf-8") as f:
        f.write("input_phrase\ttarget_phrase\n")
        f.write("paraphrase: original 1\tgenerated 1b\n") # New target for original 1
        f.write("paraphrase: original 3\tgenerated 3a\n") # New input/target pair

    combine_data_hf(original_file, selected_file, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split('\n')
    assert len(lines) == 5 # Header + 4 unique data lines
    assert lines[0] == "input_phrase\ttarget_phrase"
    # Check content (order might vary due to set, so check presence)
    expected_lines = {
        "paraphrase: original 1\toriginal 1a",
        "paraphrase: original 2\toriginal 2a",
        "paraphrase: original 1\tgenerated 1b",
        "paraphrase: original 3\tgenerated 3a"
    }
    assert set(lines[1:]) == expected_lines

def test_combine_data_hf_one_missing(clean_test_env, caplog):
    """Tests combining when the selected generated file is missing."""
    original_file = clean_test_env / "original.tsv"
    selected_file = clean_test_env / "non_existent_selected.tsv" # Missing file
    output_file = clean_test_env / "combined.tsv"

    # Create dummy original data
    with open(original_file, "w", encoding="utf-8") as f:
        f.write("input_phrase\ttarget_phrase\n")
        f.write("paraphrase: original 1\toriginal 1a\n")
        f.write("paraphrase: original 2\toriginal 2a\n")

    combine_data_hf(original_file, selected_file, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split('\n')
    assert len(lines) == 3 # Header + 2 original lines
    assert "Selected generated file does not exist" in caplog.text
    assert set(lines[1:]) == {
        "paraphrase: original 1\toriginal 1a",
        "paraphrase: original 2\toriginal 2a"
    }

# --- Tests for old/removed components (can be deleted or adapted) ---

# def test_data_preparer(clean_test_env): # Keep if DataPreparer is still used
#     """Tests the DataPreparer class."""
#     input_dir = clean_test_env / "input"
#     output_dir = clean_test_env / "output"
#     ensure_directory(input_dir)
#     # ... (rest of test) ...

# def test_run_shell_command(): # Keep if run_shell_command is still used
#     # ... (rest of test, ensure it works on target OS) ...

# def test_ensure_directory(clean_test_env): # Keep
#      new_dir = clean_test_env / "new_dir" / "subdir"
#      ensure_directory(new_dir)
#      assert new_dir.is_dir()
#      #Test idempotency
#      ensure_directory(new_dir)
#      assert new_dir.is_dir()

# def test_data_preparer_missing_input_dir(): # Keep if DataPreparer is still used
#     with pytest.raises(FileNotFoundError):
#         preparer = DataPreparer("missing_dir", "output_dir")
#         preparer.prepare()