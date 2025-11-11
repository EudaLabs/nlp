"""Tests for model evaluation framework."""
import pytest
from model_evaluation import (
    ClassificationEvaluator,
    GenerationEvaluator,
    QAEvaluator,
    NERvaluator,
)


def test_classification_evaluator():
    """Test classification evaluation."""
    evaluator = ClassificationEvaluator()
    
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    metrics = evaluator.evaluate(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_generation_evaluator():
    """Test text generation evaluation."""
    evaluator = GenerationEvaluator()
    
    predictions = ["The cat sat on the mat"]
    references = ["The cat is on the mat"]
    
    metrics = evaluator.evaluate(predictions, references, metrics=["bleu", "rouge"])
    
    assert "bleu" in metrics
    assert "rouge1" in metrics
    assert "rouge2" in metrics
    assert "rougeL" in metrics
    assert 0 <= metrics["bleu"] <= 1


def test_qa_evaluator():
    """Test QA evaluation."""
    evaluator = QAEvaluator()
    
    predictions = [{"id": "1", "prediction_text": "Paris"}]
    references = [{"id": "1", "answers": {"text": ["Paris"], "answer_start": [0]}}]
    
    metrics = evaluator.evaluate(predictions, references)
    
    assert "exact_match" in metrics
    assert "f1" in metrics
    assert metrics["exact_match"] == 100.0


def test_ner_evaluator():
    """Test NER evaluation."""
    evaluator = NERvaluator()
    
    predictions = [["B-PER", "I-PER", "O"]]
    references = [["B-PER", "I-PER", "O"]]
    
    metrics = evaluator.evaluate(predictions, references)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["f1"] == 1.0
