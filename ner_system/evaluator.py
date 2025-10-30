"""Evaluation tools for NER systems."""
from typing import Dict, List, Tuple

from sklearn.metrics import classification_report, precision_recall_fscore_support


class NEREvaluator:
    """Evaluator for NER systems."""
    
    def __init__(self, recognizer):
        """
        Initialize evaluator.
        
        Args:
            recognizer: NERRecognizer instance
        """
        self.recognizer = recognizer
    
    def evaluate(
        self,
        test_data: List[Tuple[str, Dict]],
        detailed: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate NER system on test data.
        
        Args:
            test_data: List of (text, annotations) tuples
                      annotations format: {"entities": [(start, end, label), ...]}
            detailed: Whether to return detailed metrics per entity type
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_true = []
        all_pred = []
        
        for text, annotations in test_data:
            # Get predictions
            pred_entities = self.recognizer.extract_entities(text)
            
            # Get ground truth
            true_entities = annotations.get("entities", [])
            
            # Convert to comparable format
            true_set = set(true_entities)
            pred_set = {
                (ent.start_char, ent.end_char, ent.label_)
                for ent in pred_entities
            }
            
            # Create labels for each position
            for true_ent in true_set:
                all_true.append(true_ent[2])  # Label
                if true_ent in pred_set:
                    all_pred.append(true_ent[2])
                else:
                    all_pred.append("O")  # Outside
            
            for pred_ent in pred_set:
                if pred_ent not in true_set:
                    all_true.append("O")
                    all_pred.append(pred_ent[2])
        
        # Calculate metrics
        if not all_true or not all_pred:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true,
            all_pred,
            average="weighted",
            zero_division=0,
        )
        
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        
        if detailed:
            # Add per-class metrics
            report = classification_report(
                all_true,
                all_pred,
                output_dict=True,
                zero_division=0,
            )
            metrics["detailed"] = report
        
        return metrics
    
    def evaluate_single(
        self,
        text: str,
        true_entities: List[Tuple[int, int, str]],
    ) -> Dict[str, float]:
        """
        Evaluate on a single example.
        
        Args:
            text: Input text
            true_entities: List of (start, end, label) tuples
        
        Returns:
            Dictionary with metrics
        """
        pred_entities = self.recognizer.extract_entities(text)
        
        true_set = set(true_entities)
        pred_set = {
            (ent.start_char, ent.end_char, ent.label_)
            for ent in pred_entities
        }
        
        # Calculate exact match metrics
        correct = len(true_set & pred_set)
        total_true = len(true_set)
        total_pred = len(pred_set)
        
        precision = correct / total_pred if total_pred > 0 else 0.0
        recall = correct / total_true if total_true > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "total_true": total_true,
            "total_pred": total_pred,
        }


def calculate_entity_level_metrics(
    true_entities: List[Tuple[int, int, str]],
    pred_entities: List[Tuple[int, int, str]],
) -> Dict[str, float]:
    """
    Calculate entity-level evaluation metrics.
    
    Args:
        true_entities: List of ground truth (start, end, label) tuples
        pred_entities: List of predicted (start, end, label) tuples
    
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    
    correct = len(true_set & pred_set)
    total_true = len(true_set)
    total_pred = len(pred_set)
    
    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_true if total_true > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total_true": total_true,
        "total_pred": total_pred,
    }


if __name__ == "__main__":
    # Example usage
    from recognizer import NERRecognizer
    
    # Create test data
    test_data = [
        (
            "Apple Inc. was founded by Steve Jobs.",
            {"entities": [(0, 10, "ORG"), (26, 37, "PERSON")]},
        ),
        (
            "Microsoft is based in Seattle.",
            {"entities": [(0, 9, "ORG"), (22, 29, "GPE")]},
        ),
    ]
    
    try:
        recognizer = NERRecognizer(backend="spacy")
        evaluator = NEREvaluator(recognizer)
        
        metrics = evaluator.evaluate(test_data, detailed=True)
        
        print("Evaluation Results:")
        print("=" * 50)
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        
        if "detailed" in metrics:
            print("\nDetailed Metrics:")
            print(metrics["detailed"])
    
    except Exception as e:
        print(f"Error: {e}")
