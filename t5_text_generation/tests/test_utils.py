"""
Tests for T5 text generation utilities.
"""
import pytest
from t5_text_generation.utils import (
    prepare_data,
    clean_text,
    calculate_rouge,
    calculate_bleu,
    evaluate_generation,
    truncate_text,
    count_words,
    get_summary_ratio,
    format_qa_input,
)


def test_prepare_data():
    """Test data preparation."""
    texts = ["Hello world", "Test text"]
    targets = ["Hi world", "Test output"]
    task_prefix = "paraphrase: "

    data = prepare_data(texts, targets, task_prefix)

    assert len(data) == 2
    assert data[0]["input"] == "paraphrase: Hello world"
    assert data[0]["output"] == "Hi world"
    assert data[1]["input"] == "paraphrase: Test text"
    assert data[1]["output"] == "Test output"


def test_prepare_data_mismatch():
    """Test data preparation with mismatched lengths."""
    texts = ["Hello world"]
    targets = ["Hi world", "Extra target"]

    with pytest.raises(ValueError):
        prepare_data(texts, targets)


def test_clean_text():
    """Test text cleaning."""
    text = "  Hello   world  \n  with   extra   spaces  "
    cleaned = clean_text(text)
    assert cleaned == "Hello world with extra spaces"


def test_clean_text_empty():
    """Test cleaning empty text."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_calculate_rouge():
    """Test ROUGE score calculation."""
    predictions = ["The cat sat on the mat"]
    references = ["The cat is on the mat"]

    scores = calculate_rouge(predictions, references)

    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert 0 <= scores["rouge1"] <= 1
    assert 0 <= scores["rouge2"] <= 1
    assert 0 <= scores["rougeL"] <= 1


def test_calculate_bleu():
    """Test BLEU score calculation."""
    predictions = ["The cat sat on the mat"]
    references = ["The cat is on the mat"]

    score = calculate_bleu(predictions, references)

    assert 0 <= score <= 1
    assert isinstance(score, float)


def test_evaluate_generation():
    """Test generation evaluation with multiple metrics."""
    predictions = ["The cat sat on the mat"]
    references = ["The cat is on the mat"]

    results = evaluate_generation(predictions, references, metrics=["rouge", "bleu"])

    assert "rouge1" in results
    assert "rouge2" in results
    assert "rougeL" in results
    assert "bleu" in results


def test_evaluate_generation_rouge_only():
    """Test evaluation with ROUGE only."""
    predictions = ["Test prediction"]
    references = ["Test reference"]

    results = evaluate_generation(predictions, references, metrics=["rouge"])

    assert "rouge1" in results
    assert "bleu" not in results


def test_truncate_text():
    """Test text truncation."""
    text = " ".join(["word"] * 150)
    truncated = truncate_text(text, max_words=100)

    word_count = len(truncated.replace("...", "").split())
    assert word_count == 100
    assert truncated.endswith("...")


def test_truncate_text_short():
    """Test truncation with short text."""
    text = "Short text"
    truncated = truncate_text(text, max_words=100)
    assert truncated == text
    assert not truncated.endswith("...")


def test_count_words():
    """Test word counting."""
    assert count_words("Hello world") == 2
    assert count_words("One") == 1
    assert count_words("") == 1  # Empty string splits to ['']
    assert count_words("Multiple words in a sentence") == 5


def test_get_summary_ratio():
    """Test summary compression ratio calculation."""
    original = "This is a long original text with many words"
    summary = "Short summary"

    ratio = get_summary_ratio(original, summary)

    assert 0 < ratio < 1  # Summary should be shorter
    assert isinstance(ratio, float)


def test_get_summary_ratio_empty_original():
    """Test ratio with empty original text."""
    ratio = get_summary_ratio("", "summary")
    assert ratio == 0.0


def test_format_qa_input():
    """Test QA input formatting."""
    question = "What is AI?"
    context = "AI stands for Artificial Intelligence"

    formatted = format_qa_input(question, context)

    assert formatted == "question: What is AI? context: AI stands for Artificial Intelligence"
    assert "question:" in formatted
    assert "context:" in formatted


def test_format_qa_input_empty():
    """Test QA formatting with empty strings."""
    formatted = format_qa_input("", "")
    assert formatted == "question:  context: "
