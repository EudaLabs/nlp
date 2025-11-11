"""
T5 Text Generation Project

This package provides utilities for text generation using T5 models,
including summarization, paraphrasing, translation, and custom text generation tasks.
"""

from .config import T5Config
from .inference import T5Generator
from .train import T5Trainer
from .utils import prepare_data, evaluate_generation

__version__ = "1.0.0"
__all__ = [
    "T5Config",
    "T5Generator",
    "T5Trainer",
    "prepare_data",
    "evaluate_generation",
]
