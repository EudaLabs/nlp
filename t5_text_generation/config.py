"""
Configuration classes for T5 text generation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class T5Config:
    """Configuration for T5 text generation."""

    # Model settings
    model_name: str = "t5-base"
    cache_dir: Optional[str] = None
    device: str = "cpu"

    # Generation settings
    max_length: int = 512
    min_length: int = 10
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0

    # Training settings
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500

    # Task-specific settings
    summarization_max_length: int = 150
    summarization_min_length: int = 50
    translation_max_length: int = 256
    paraphrase_max_length: int = 256

    def __post_init__(self):
        """Validate configuration."""
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than min_length")
        if self.num_beams < 1:
            raise ValueError("num_beams must be at least 1")
        if not 0 < self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")


@dataclass
class TaskConfig:
    """Task-specific configuration."""

    task_type: str  # summarize, translate, paraphrase, qa
    prefix: str
    max_length: int = 256
    min_length: int = 10
    num_beams: int = 4

    @classmethod
    def summarization(cls, max_length: int = 150, min_length: int = 50):
        """Create configuration for summarization task."""
        return cls(
            task_type="summarization",
            prefix="summarize: ",
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
        )

    @classmethod
    def paraphrase(cls, max_length: int = 256):
        """Create configuration for paraphrasing task."""
        return cls(
            task_type="paraphrase",
            prefix="paraphrase: ",
            max_length=max_length,
            min_length=10,
            num_beams=5,
        )

    @classmethod
    def translation(cls, source_lang: str, target_lang: str, max_length: int = 256):
        """Create configuration for translation task."""
        return cls(
            task_type="translation",
            prefix=f"translate {source_lang} to {target_lang}: ",
            max_length=max_length,
            min_length=10,
            num_beams=4,
        )

    @classmethod
    def question_answering(cls, max_length: int = 128):
        """Create configuration for question answering task."""
        return cls(
            task_type="qa",
            prefix="question: ",
            max_length=max_length,
            min_length=5,
            num_beams=4,
        )
