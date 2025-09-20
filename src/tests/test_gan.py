import pytest
import pandas as pd
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.utils import (
    gemini_generate_paraphrase,
    gemini_classify_paraphrase,
    setup_logger,
    load_gemini_api_key,
    generate_mock_paraphrases,
    postprocess_discriminator_output_gemini,
    ensure_directory
)
from src.config import CONFIG
from src.plugins import get_plugin_manager


class DummyClient:
    """Mock client for testing Gemini API calls."""
    class ModelsNS:
        def __init__(self, response_text: str):
            self._text = response_text
        def generate_content(self, model: str, contents: str):
            # mimic google-genai response shape with candidates[0].content.parts[0].text and .text
            part = SimpleNamespace(text=self._text)
            content = SimpleNamespace(parts=[part])
            candidate = SimpleNamespace(content=content)
            return SimpleNamespace(candidates=[candidate], text=self._text)
    def __init__(self, response_text: str):
        self.models = DummyClient.ModelsNS(response_text)


def test_utils_generation_and_classification_mocked():
    """Test basic generation and classification with mocked client."""
    client = DummyClient("human")  # will return "human" for both calls
    model_name = "gemini-2.5-pro"

    # Test generation
    gen = gemini_generate_paraphrase(
        input_text="hello world",
        client=client,
        model_name=model_name,
        prompt_template="Paraphrase: {text}",
        max_retries=1,
        delay=0,
    )
    assert isinstance(gen, str) and len(gen) > 0

    # Test classification
    cls = gemini_classify_paraphrase(
        text="anything",
        client=client,
        model_name=model_name,
        prompt_template="Classify: {text}",
        max_retries=1,
        delay=0,
    )
    assert cls == "human"


def test_setup_logger():
    """Test logger setup functionality."""
    with patch('src.utils.logging') as mock_logging:
        mock_handler = Mock()
        mock_logging.StreamHandler.return_value = mock_handler
        mock_logging.FileHandler.return_value = mock_handler
        mock_logger = Mock()
        mock_logging.getLogger.return_value = mock_logger

        logger = setup_logger('/tmp/test_logs', CONFIG)

        assert logger is not None
        mock_logging.getLogger.assert_called_once()


def test_generate_mock_paraphrases():
    """Test mock data generation."""
    data = generate_mock_paraphrases(10)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 10
    assert 'input_text' in data.columns
    assert data['input_text'].nunique() == 10  # Should have unique entries


def test_postprocess_discriminator_output_gemini():
    """Test post-processing of classification results."""
    test_data = [
        {
            'input_text': 'test phrase 1',
            'generated_text': 'paraphrase 1',
            'classification': 'human'
        },
        {
            'input_text': 'test phrase 2',
            'generated_text': 'paraphrase 2',
            'classification': 'machine'
        },
        {
            'input_text': 'test phrase 3',
            'generated_text': 'paraphrase 3',
            'classification': 'error'
        }
    ]

    result = postprocess_discriminator_output_gemini(test_data)

    assert len(result) == 1  # Only human classification should be selected
    assert result[0]['input_text'] == 'test phrase 1'
    assert result[0]['target_text'] == 'paraphrase 1'
    assert result[0]['classification'] == 'human'


def test_ensure_directory():
    """Test directory creation utility."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, 'test', 'nested', 'dir')

        # Should not raise an exception
        ensure_directory(test_dir)

        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)


def test_plugin_system():
    """Test plugin system functionality."""
    plugin_manager = get_plugin_manager()

    # Test that plugin manager exists
    assert plugin_manager is not None

    # Test plugin discovery
    plugins = plugin_manager.discover_plugins()
    assert isinstance(plugins, list)

    # Test loading a specific plugin (if system_admin plugin exists)
    if 'system_admin' in plugins:
        success = plugin_manager.load_plugin('system_admin')
        if success:
            plugin = plugin_manager.get_plugin('system_admin')
            assert plugin is not None


def test_api_error_handling():
    """Test API error handling and retries."""
    # Test with a client that always fails
    failing_client = DummyClient("error")
    failing_client.models.generate_content = Mock(side_effect=Exception("API Error"))

    result = gemini_generate_paraphrase(
        input_text="test",
        client=failing_client,
        model_name="gemini-2.5-pro",
        prompt_template="Paraphrase: {text}",
        max_retries=2,
        delay=0,
    )

    # Should return None after retries
    assert result is None


def test_classification_edge_cases():
    """Test classification with various edge cases."""
    test_cases = [
        ("This is clearly human writing", "human"),
        ("This looks like machine output", "machine"),
        ("Random text without clear indicators", "error"),
        ("", "error"),
        ("   ", "error")
    ]

    for text, expected in test_cases:
        if expected != "error":
            client = DummyClient(expected)
            result = gemini_classify_paraphrase(
                text=text,
                client=client,
                model_name="gemini-2.5-pro",
                prompt_template="Classify: {text}",
                max_retries=1,
                delay=0,
            )
            assert result == expected