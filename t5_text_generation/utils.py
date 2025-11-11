"""
Utility functions for T5 text generation.
"""
from typing import List, Dict, Any
import re
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk


def prepare_data(
    texts: List[str],
    targets: List[str],
    task_prefix: str = "summarize: ",
) -> List[Dict[str, str]]:
    """
    Prepare data for T5 training.

    Args:
        texts: List of input texts
        targets: List of target outputs
        task_prefix: Task prefix (e.g., "summarize: ", "paraphrase: ")

    Returns:
        List of dicts with 'input' and 'output' keys
    """
    if len(texts) != len(targets):
        raise ValueError("texts and targets must have the same length")

    return [
        {"input": f"{task_prefix}{text}", "output": target}
        for text, target in zip(texts, targets)
    ]


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dict of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Average BLEU score
    """
    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    smoothing = SmoothingFunction().method1
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)


def evaluate_generation(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ["rouge", "bleu"],
) -> Dict[str, Any]:
    """
    Evaluate generated text using multiple metrics.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        metrics: List of metrics to calculate

    Returns:
        Dict of evaluation metrics
    """
    results = {}

    if "rouge" in metrics:
        rouge_scores = calculate_rouge(predictions, references)
        results.update(rouge_scores)

    if "bleu" in metrics:
        bleu_score = calculate_bleu(predictions, references)
        results["bleu"] = bleu_score

    return results


def truncate_text(text: str, max_words: int = 100) -> str:
    """
    Truncate text to maximum number of words.

    Args:
        text: Input text
        max_words: Maximum number of words

    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    return nltk.sent_tokenize(text)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def get_summary_ratio(original: str, summary: str) -> float:
    """
    Calculate compression ratio between original and summary.

    Args:
        original: Original text
        summary: Summary text

    Returns:
        Compression ratio (summary_length / original_length)
    """
    original_words = count_words(original)
    summary_words = count_words(summary)

    if original_words == 0:
        return 0.0

    return summary_words / original_words


def format_qa_input(question: str, context: str) -> str:
    """
    Format question and context for T5 input.

    Args:
        question: Question text
        context: Context text

    Returns:
        Formatted input string
    """
    return f"question: {question} context: {context}"
