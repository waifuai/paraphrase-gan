"""
Test Configuration Module

This module contains pytest configuration and fixtures for the test suite.
It provides shared test data, directory fixtures, and common test utilities
that can be used across multiple test files.

The fixtures include test data for input phrases and expected outputs, as well
as clean test environments for isolated testing of components.
"""

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