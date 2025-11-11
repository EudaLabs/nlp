"""
Configuration classes for GPT-2 fine-tuning.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT2Config:
    """Configuration for GPT-2 text generation."""

    # Model settings
    model_name: str = "gpt2"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl
    cache_dir: Optional[str] = None
    device: str = "cpu"

    # Generation settings
    max_length: int = 100
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False

    # Training settings
    learning_rate: float = 5e-5
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    block_size: int = 512  # Maximum sequence length

    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")

        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than min_length")

        if not 0 < self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")

        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 100
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    @classmethod
    def creative(cls):
        """Configuration for creative generation."""
        return cls(
            temperature=0.9,
            top_k=40,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
        )

    @classmethod
    def focused(cls):
        """Configuration for focused, deterministic generation."""
        return cls(
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    @classmethod
    def balanced(cls):
        """Configuration for balanced generation."""
        return cls(
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.15,
        )
