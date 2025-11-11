"""Question Answering evaluation metrics."""
from typing import List, Dict
import string
import re
from collections import Counter


class QAEvaluator:
    """Evaluator for Question Answering tasks."""

    def evaluate(
        self,
        predictions: List[Dict],
        references: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate QA predictions.

        Args:
            predictions: List of prediction dicts with 'id' and 'prediction_text'
            references: List of reference dicts with 'id' and 'answers'

        Returns:
            Dict with 'exact_match' and 'f1' scores
        """
        exact_matches = []
        f1_scores = []

        pred_dict = {p["id"]: p["prediction_text"] for p in predictions}
        ref_dict = {r["id"]: r["answers"] for r in references}

        for qid in ref_dict:
            if qid not in pred_dict:
                exact_matches.append(0)
                f1_scores.append(0)
                continue

            prediction = pred_dict[qid]
            ground_truths = ref_dict[qid]["text"]

            exact_match = max(
                self._exact_match_score(prediction, gt) for gt in ground_truths
            )
            f1 = max(self._f1_score(prediction, gt) for gt in ground_truths)

            exact_matches.append(exact_match)
            f1_scores.append(f1)

        return {
            "exact_match": sum(exact_matches) / len(exact_matches) * 100,
            "f1": sum(f1_scores) / len(f1_scores) * 100,
        }

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer string."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _exact_match_score(self, prediction: str, ground_truth: str) -> int:
        """Calculate exact match score."""
        return int(
            self._normalize_answer(prediction) == self._normalize_answer(ground_truth)
        )

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score."""
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1
