"""
GPT-2 Fine-tuning Project

This package provides utilities for fine-tuning GPT-2 models
for text generation and completion tasks.
"""

from .config import GPT2Config
from .inference import GPT2Generator
from .train import GPT2Trainer
from .utils import prepare_text_data, PromptTemplate

__version__ = "1.0.0"
__all__ = [
    "GPT2Config",
    "GPT2Generator",
    "GPT2Trainer",
    "prepare_text_data",
    "PromptTemplate",
]
