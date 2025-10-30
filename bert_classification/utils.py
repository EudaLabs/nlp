"""Utility functions for BERT text classification."""
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
    
    Returns:
        Dictionary with accuracy, F1, precision, and recall
    """
    accuracy = accuracy_score(labels, predictions)
    
    # Determine if binary or multi-class
    num_classes = len(np.unique(labels))
    average = "binary" if num_classes == 2 else "macro"
    
    f1 = f1_score(labels, predictions, average=average)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def save_metrics(metrics: Dict[str, float], output_path: str):
    """Save metrics to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict[str, float]:
    """Load metrics from a JSON file."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def print_training_summary(
    num_train_samples: int,
    num_val_samples: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Print a summary of training configuration."""
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Number of training samples: {num_train_samples:,}")
    logger.info(f"Number of validation samples: {num_val_samples:,}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Total training steps: {(num_train_samples // batch_size) * epochs}")
    logger.info("=" * 60)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


class EarlyStopping:
    """Early stopping callback to stop training when metric stops improving."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "min"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
