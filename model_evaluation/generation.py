"""Text generation evaluation metrics."""
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from typing import List, Dict
import nltk


class GenerationEvaluator:
    """Evaluator for text generation tasks."""

    def __init__(self):
        """Initialize evaluator."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str] = ["bleu", "rouge"],
    ) -> Dict[str, float]:
        """
        Evaluate text generation.

        Args:
            predictions: Generated texts
            references: Reference texts
            metrics: List of metrics to compute

        Returns:
            Dict of metric scores
        """
        results = {}

        if "bleu" in metrics:
            results["bleu"] = self.calculate_bleu(predictions, references)

        if "rouge" in metrics:
            rouge_scores = self.calculate_rouge(predictions, references)
            results.update(rouge_scores)

        return results

    def calculate_bleu(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate BLEU score."""
        smoothing = SmoothingFunction().method1
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def calculate_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        rouge1, rouge2, rougeL = [], [], []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougeL.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": sum(rouge1) / len(rouge1) if rouge1 else 0.0,
            "rouge2": sum(rouge2) / len(rouge2) if rouge2 else 0.0,
            "rougeL": sum(rougeL) / len(rougeL) if rougeL else 0.0,
        }
