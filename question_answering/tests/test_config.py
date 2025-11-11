"""Tests for Question Answering system."""
import pytest
from question_answering.config import QAConfig


def test_qa_config_defaults():
    """Test QAConfig defaults."""
    config = QAConfig()
    
    assert config.model_name == "distilbert-base-cased-distilled-squad"
    assert config.device == "cpu"
    assert config.max_length == 384
    assert config.top_k == 1


def test_qa_config_custom():
    """Test QAConfig with custom values."""
    config = QAConfig(
        model_name="bert-base-uncased",
        device="cuda",
        max_length=512,
        top_k=3
    )
    
    assert config.model_name == "bert-base-uncased"
    assert config.device == "cuda"
    assert config.max_length == 512
    assert config.top_k == 3
