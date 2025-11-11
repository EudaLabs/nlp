"""Model Evaluation Framework Package."""

from .classification import ClassificationEvaluator
from .generation import GenerationEvaluator
from .qa_metrics import QAEvaluator
from .ner_metrics import NERvaluator
from .visualization import plot_confusion_matrix, plot_roc_curve

__version__ = "1.0.0"
__all__ = [
    "ClassificationEvaluator",
    "GenerationEvaluator",
    "QAEvaluator",
    "NERvaluator",
    "plot_confusion_matrix",
    "plot_roc_curve",
]
