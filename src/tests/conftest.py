import pytest
from pathlib import Path
from src.utils import ensure_directory

@pytest.fixture(scope="module")
def test_data():
    return {
        "input_texts": ["sample phrase 1", "sample phrase 2"],
        "expected_outputs": ["paraphrase 1", "paraphrase 2"],
    }

@pytest.fixture
def clean_test_env(tmp_path: Path):
    test_dir = tmp_path / "gan_test"
    ensure_directory(test_dir)
    yield test_dir
    # pytest auto-cleanup handles tmp_path