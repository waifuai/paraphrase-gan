import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def test_data():
    return {
        "input_texts": ["sample phrase 1", "sample phrase 2"],
        "expected_outputs": ["paraphrase 1", "paraphrase 2"]
    }

@pytest.fixture
def clean_test_env(tmp_path):
    test_dir = tmp_path / "gan_test"
    test_dir.mkdir()
    yield test_dir
    # Auto-cleanup by pytest

def run_training_command():
    # Placeholder for running training command
    pass