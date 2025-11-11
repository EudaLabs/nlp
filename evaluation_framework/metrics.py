"""
Comprehensive metrics for NLP model evaluation.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report as sklearn_classification_report
)
from collections import Counter
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Metrics for text classification tasks."""
    
    @staticmethod
    def accuracy(y_true: List, y_pred: List) -> float:
        """Calculate accuracy."""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def precision_recall_f1(
        y_true: List,
        y_pred: List,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('micro', 'macro', 'weighted', 'binary')
            
        Returns:
            Dictionary with precision, recall, and f1
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    @staticmethod
    def confusion_matrix(y_true: List, y_pred: List) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def classification_report(
        y_true: List,
        y_pred: List,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Generate detailed classification report."""
        return sklearn_classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
    
    @staticmethod
    def evaluate_all(
        y_true: List,
        y_pred: List,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
            **ClassificationMetrics.precision_recall_f1(y_true, y_pred),
            'confusion_matrix': ClassificationMetrics.confusion_matrix(y_true, y_pred).tolist(),
            'report': ClassificationMetrics.classification_report(y_true, y_pred, target_names)
        }
        
        return metrics


class QAMetrics:
    """Metrics for Question Answering tasks."""
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def exact_match(pred: str, truth: str) -> float:
        """
        Calculate exact match score.
        
        Returns:
            1.0 if normalized texts match, 0.0 otherwise
        """
        return float(QAMetrics.normalize_answer(pred) == QAMetrics.normalize_answer(truth))
    
    @staticmethod
    def f1_score(pred: str, truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_tokens = QAMetrics.normalize_answer(pred).split()
        truth_tokens = QAMetrics.normalize_answer(truth).split()
        
        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate batch of QA predictions.
        
        Returns:
            Dictionary with EM and F1 scores
        """
        em_scores = [
            QAMetrics.exact_match(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        
        f1_scores = [
            QAMetrics.f1_score(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        
        return {
            'exact_match': np.mean(em_scores),
            'f1': np.mean(f1_scores),
            'total': len(predictions)
        }


class GenerationMetrics:
    """Metrics for text generation tasks."""
    
    @staticmethod
    def bleu_score(
        predictions: List[str],
        references: List[List[str]],
        max_order: int = 4
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.
        
        Note: This is a simple implementation. For production,
        use the 'evaluate' library or sacrebleu.
        
        Args:
            predictions: List of predicted texts
            references: List of reference text lists
            max_order: Maximum n-gram order
            
        Returns:
            Dictionary with BLEU scores
        """
        try:
            from evaluate import load
            bleu = load("bleu")
            results = bleu.compute(
                predictions=predictions,
                references=references,
                max_order=max_order
            )
            return results
        except ImportError:
            logger.warning("evaluate library not available. Install with: pip install evaluate")
            return {"bleu": 0.0}
    
    @staticmethod
    def rouge_score(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            from evaluate import load
            rouge = load("rouge")
            results = rouge.compute(
                predictions=predictions,
                references=references
            )
            return results
        except ImportError:
            logger.warning("evaluate library not available. Install with: pip install evaluate")
            return {}
    
    @staticmethod
    def meteor_score(
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Calculate METEOR score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            METEOR score
        """
        try:
            from evaluate import load
            meteor = load("meteor")
            results = meteor.compute(
                predictions=predictions,
                references=references
            )
            return results['meteor']
        except ImportError:
            logger.warning("evaluate library not available. Install with: pip install evaluate")
            return 0.0
    
    @staticmethod
    def evaluate_all(
        predictions: List[str],
        references: Union[List[str], List[List[str]]]
    ) -> Dict[str, Any]:
        """
        Comprehensive generation evaluation.
        
        Args:
            predictions: Predicted texts
            references: Reference texts (single or multiple per prediction)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # ROUGE (single reference)
        if isinstance(references[0], str):
            metrics['rouge'] = GenerationMetrics.rouge_score(predictions, references)
            metrics['meteor'] = GenerationMetrics.meteor_score(predictions, references)
        
        # BLEU (multiple references)
        if isinstance(references[0], list):
            metrics['bleu'] = GenerationMetrics.bleu_score(predictions, references)
        
        return metrics


class PerplexityMetric:
    """Perplexity metric for language models."""
    
    @staticmethod
    def calculate(loss: float) -> float:
        """
        Calculate perplexity from loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity value
        """
        import math
        return math.exp(loss)
    
    @staticmethod
    def calculate_batch(losses: List[float]) -> float:
        """
        Calculate average perplexity from batch of losses.
        
        Args:
            losses: List of cross-entropy losses
            
        Returns:
            Average perplexity
        """
        import math
        avg_loss = np.mean(losses)
        return math.exp(avg_loss)


class ModelEvaluator:
    """Unified evaluator for different NLP tasks."""
    
    def __init__(self, task: str):
        """
        Initialize evaluator.
        
        Args:
            task: Task type ('classification', 'qa', 'generation', 'language_modeling')
        """
        self.task = task
        
        if task == 'classification':
            self.metrics = ClassificationMetrics()
        elif task == 'qa':
            self.metrics = QAMetrics()
        elif task == 'generation':
            self.metrics = GenerationMetrics()
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def evaluate(
        self,
        predictions: List,
        references: List,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate predictions.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional task-specific parameters
            
        Returns:
            Evaluation metrics
        """
        if self.task == 'classification':
            return self.metrics.evaluate_all(references, predictions, **kwargs)
        elif self.task == 'qa':
            return self.metrics.evaluate_batch(predictions, references)
        elif self.task == 'generation':
            return self.metrics.evaluate_all(predictions, references)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def print_report(self, metrics: Dict[str, Any]):
        """Print formatted evaluation report."""
        print("\n" + "="*80)
        print(f"{self.task.upper()} EVALUATION REPORT")
        print("="*80)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, str):
                print(f"\n{key}:\n{value}")
            elif isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
        
        print("="*80 + "\n")


def evaluate_classification(y_true: List, y_pred: List, **kwargs) -> Dict[str, Any]:
    """Convenience function for classification evaluation."""
    return ClassificationMetrics.evaluate_all(y_true, y_pred, **kwargs)


def evaluate_qa(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Convenience function for QA evaluation."""
    return QAMetrics.evaluate_batch(predictions, references)


def evaluate_generation(
    predictions: List[str],
    references: Union[List[str], List[List[str]]]
) -> Dict[str, Any]:
    """Convenience function for generation evaluation."""
    return GenerationMetrics.evaluate_all(predictions, references)
