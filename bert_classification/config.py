"""Configuration classes for BERT text classification."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the BERT model."""
    
    model_name: str = "bert-base-uncased"
    num_classes: int = 2
    max_length: int = 512
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False  # Mixed precision training
    
    # Checkpointing
    output_dir: str = "./models/bert-classifier"
    save_steps: int = 500
    save_total_limit: int = 2
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_steps: int = 100
    logging_dir: Optional[str] = None
    
    # Data
    train_test_split: float = 0.8
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for dataset."""
    
    dataset_name: Optional[str] = None  # e.g., "imdb"
    text_column: str = "text"
    label_column: str = "label"
    
    # Custom data paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Data processing
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
