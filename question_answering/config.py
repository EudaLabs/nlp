"""Configuration for QA system."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class QAConfig:
    """Question Answering configuration."""
    
    model_name: str = "distilbert-base-cased-distilled-squad"
    device: str = "cpu"
    max_length: int = 384
    max_answer_length: int = 30
    doc_stride: int = 128
    top_k: int = 1
    min_confidence: float = 0.0
    
    # Training
    learning_rate: float = 3e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
