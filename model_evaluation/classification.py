"""Classification evaluation metrics."""
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
)
from typing import List, Dict, Optional
import numpy as np


class ClassificationEvaluator:
    """Evaluator for classification tasks."""

    def evaluate(
        self,
        y_true: List,
        y_pred: List,
        y_prob: Optional[List] = None,
        average: str = "weighted",
        labels: Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Evaluate classification predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for ROC-AUC)
            average: Averaging strategy for multi-class
            labels: List of label names

        Returns:
            Dict of evaluation metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        # Matthews Correlation Coefficient
        try:
            metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics["mcc"] = 0.0

        # ROC-AUC (if probabilities provided)
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, average=average, multi_class="ovr"
                )
            except:
                pass

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        return metrics

    def detailed_report(
        self,
        y_true: List,
        y_pred: List,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred, target_names=target_names)

    def per_class_metrics(
        self, y_true: List, y_pred: List, labels: List[str]
    ) -> Dict[str, Dict]:
        """Calculate metrics per class."""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class = {}
        for i, label in enumerate(labels):
            per_class[label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            }

        return per_class
