"""
Tests for T5 configuration.
"""
import pytest
from t5_text_generation.config import T5Config, TaskConfig


def test_t5_config_defaults():
    """Test T5Config with default values."""
    config = T5Config()

    assert config.model_name == "t5-base"
    assert config.device == "cpu"
    assert config.max_length == 512
    assert config.min_length == 10
    assert config.num_beams == 4
    assert config.learning_rate == 5e-5


def test_t5_config_custom():
    """Test T5Config with custom values."""
    config = T5Config(
        model_name="t5-small",
        device="cuda",
        max_length=256,
        num_beams=5,
        learning_rate=1e-4,
    )

    assert config.model_name == "t5-small"
    assert config.device == "cuda"
    assert config.max_length == 256
    assert config.num_beams == 5
    assert config.learning_rate == 1e-4


def test_t5_config_validation_max_min_length():
    """Test validation of max_length and min_length."""
    with pytest.raises(ValueError, match="max_length must be greater than min_length"):
        T5Config(max_length=10, min_length=20)


def test_t5_config_validation_num_beams():
    """Test validation of num_beams."""
    with pytest.raises(ValueError, match="num_beams must be at least 1"):
        T5Config(num_beams=0)


def test_t5_config_validation_temperature():
    """Test validation of temperature."""
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        T5Config(temperature=3.0)

    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        T5Config(temperature=0.0)


def test_t5_config_validation_learning_rate():
    """Test validation of learning_rate."""
    with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
        T5Config(learning_rate=1.5)

    with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
        T5Config(learning_rate=0.0)


def test_task_config_summarization():
    """Test TaskConfig for summarization."""
    config = TaskConfig.summarization(max_length=200, min_length=40)

    assert config.task_type == "summarization"
    assert config.prefix == "summarize: "
    assert config.max_length == 200
    assert config.min_length == 40
    assert config.num_beams == 4


def test_task_config_paraphrase():
    """Test TaskConfig for paraphrasing."""
    config = TaskConfig.paraphrase(max_length=300)

    assert config.task_type == "paraphrase"
    assert config.prefix == "paraphrase: "
    assert config.max_length == 300
    assert config.min_length == 10
    assert config.num_beams == 5


def test_task_config_translation():
    """Test TaskConfig for translation."""
    config = TaskConfig.translation("English", "French", max_length=256)

    assert config.task_type == "translation"
    assert config.prefix == "translate English to French: "
    assert config.max_length == 256
    assert config.min_length == 10
    assert config.num_beams == 4


def test_task_config_qa():
    """Test TaskConfig for question answering."""
    config = TaskConfig.question_answering(max_length=100)

    assert config.task_type == "qa"
    assert config.prefix == "question: "
    assert config.max_length == 100
    assert config.min_length == 5
    assert config.num_beams == 4


def test_task_config_defaults():
    """Test TaskConfig with default parameters."""
    config = TaskConfig.summarization()

    assert config.max_length == 150
    assert config.min_length == 50


def test_task_config_translation_languages():
    """Test translation with different language pairs."""
    configs = [
        TaskConfig.translation("English", "German"),
        TaskConfig.translation("French", "Spanish"),
        TaskConfig.translation("German", "English"),
    ]

    assert "English to German" in configs[0].prefix
    assert "French to Spanish" in configs[1].prefix
    assert "German to English" in configs[2].prefix
