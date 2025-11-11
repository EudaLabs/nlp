"""NER evaluation metrics."""
from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import List, Dict


class NERvaluator:
    """Evaluator for Named Entity Recognition tasks."""

    def evaluate(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate NER predictions.

        Args:
            predictions: List of predicted tag sequences
            references: List of reference tag sequences

        Returns:
            Dict of evaluation metrics
        """
        # Flatten sequences
        y_pred = [tag for seq in predictions for tag in seq]
        y_true = [tag for seq in references for tag in seq]

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def detailed_report(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> str:
        """Generate detailed classification report."""
        y_pred = [tag for seq in predictions for tag in seq]
        y_true = [tag for seq in references for tag in seq]

        return classification_report(y_true, y_pred, zero_division=0)
