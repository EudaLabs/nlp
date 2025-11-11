# Model Evaluation Framework

Comprehensive evaluation utilities for NLP models with support for various metrics and datasets.

## 🎯 Features

- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Generation Metrics**: BLEU, ROUGE, METEOR, Perplexity
- **QA Metrics**: Exact Match, F1 Score
- **NER Metrics**: Token-level and entity-level evaluation
- **Confusion Matrix**: Visualization and analysis
- **Cross-validation**: K-fold evaluation support
- **Benchmark Datasets**: Easy loading and evaluation

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Classification Evaluation

```python
from model_evaluation import ClassificationEvaluator

evaluator = ClassificationEvaluator()

# Evaluate predictions
metrics = evaluator.evaluate(
    y_true=[0, 1, 1, 0, 1],
    y_pred=[0, 1, 0, 0, 1]
)

print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
```

### Text Generation Evaluation

```python
from model_evaluation import GenerationEvaluator

evaluator = GenerationEvaluator()

references = ["The cat sat on the mat"]
predictions = ["The cat is on the mat"]

metrics = evaluator.evaluate(
    predictions=predictions,
    references=references,
    metrics=["bleu", "rouge"]
)

print(f"BLEU: {metrics['bleu']:.4f}")
print(f"ROUGE-1: {metrics['rouge1']:.4f}")
print(f"ROUGE-L: {metrics['rougeL']:.4f}")
```

### Question Answering Evaluation

```python
from model_evaluation import QAEvaluator

evaluator = QAEvaluator()

predictions = [{"prediction_text": "Paris", "id": "1"}]
references = [{"answers": {"text": ["Paris"], "answer_start": [0]}, "id": "1"}]

metrics = evaluator.evaluate(predictions, references)

print(f"Exact Match: {metrics['exact_match']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")
```

### NER Evaluation

```python
from model_evaluation import NERvaluator

evaluator = NERvaluator()

predictions = [["B-PER", "I-PER", "O", "B-LOC"]]
references = [["B-PER", "I-PER", "O", "B-LOC"]]

metrics = evaluator.evaluate(predictions, references)

print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1: {metrics['f1']:.2f}")
```

## 📊 Visualization

### Confusion Matrix

```python
from model_evaluation import plot_confusion_matrix

plot_confusion_matrix(
    y_true=[0, 1, 2, 0, 1, 2],
    y_pred=[0, 2, 2, 0, 1, 1],
    labels=["Class A", "Class B", "Class C"],
    save_path="confusion_matrix.png"
)
```

### ROC Curve

```python
from model_evaluation import plot_roc_curve

plot_roc_curve(
    y_true=[0, 1, 1, 0, 1],
    y_scores=[0.1, 0.9, 0.8, 0.3, 0.7],
    save_path="roc_curve.png"
)
```

## 🎨 Advanced Features

### Cross-Validation

```python
from model_evaluation import cross_validate

results = cross_validate(
    model=my_model,
    X=features,
    y=labels,
    cv=5,
    metrics=["accuracy", "f1"]
)

print(f"Mean Accuracy: {results['accuracy_mean']:.2f}")
print(f"Std Accuracy: {results['accuracy_std']:.2f}")
```

### Model Comparison

```python
from model_evaluation import ModelComparator

comparator = ModelComparator()

comparator.add_model("Model A", predictions_a, references)
comparator.add_model("Model B", predictions_b, references)

comparison = comparator.compare(metrics=["accuracy", "f1", "precision"])
comparison.plot(save_path="model_comparison.png")
```

## 📈 Supported Metrics

### Classification
- Accuracy
- Precision (micro, macro, weighted)
- Recall (micro, macro, weighted)
- F1 Score (micro, macro, weighted)
- ROC-AUC
- Matthews Correlation Coefficient

### Text Generation
- BLEU (1-4 gram)
- ROUGE (1, 2, L)
- METEOR
- Perplexity
- BERTScore

### Question Answering
- Exact Match
- F1 Score

### Named Entity Recognition
- Token-level metrics
- Entity-level metrics
- Precision, Recall, F1 per entity type

## 🧪 Testing

```bash
pytest tests/ -v --cov=model_evaluation
```

## 📚 Examples

- `classification_demo.py` - Classification evaluation
- `generation_demo.py` - Text generation metrics
- `qa_demo.py` - Question answering evaluation
- `ner_demo.py` - NER evaluation
- `comparison_demo.py` - Model comparison

## 🤝 Contributing

Contributions welcome! Please include tests for new metrics.

## 📄 License

MIT License

## 🔗 References

- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)
