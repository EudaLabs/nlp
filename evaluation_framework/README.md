# NLP Evaluation Framework

Comprehensive evaluation metrics and tools for various NLP tasks.

## 🎯 Features

- **Classification Metrics** - Accuracy, precision, recall, F1, confusion matrix
- **QA Metrics** - Exact Match (EM), token-level F1
- **Generation Metrics** - BLEU, ROUGE, METEOR scores
- **Perplexity** - Language model evaluation
- **Unified Interface** - Single API for all tasks
- **Easy Integration** - Works with any model output

## 🚀 Quick Start

### Classification Evaluation

```python
from evaluation_framework.metrics import evaluate_classification

y_true = ["positive", "negative", "neutral", "positive"]
y_pred = ["positive", "negative", "positive", "positive"]

metrics = evaluate_classification(y_true, y_pred)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"\n{metrics['report']}")
```

### Question Answering Evaluation

```python
from evaluation_framework.metrics import evaluate_qa

predictions = ["Paris", "1991", "Python"]
references = ["Paris, France", "1991", "Python programming language"]

metrics = evaluate_qa(predictions, references)

print(f"Exact Match: {metrics['exact_match']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### Text Generation Evaluation

```python
from evaluation_framework.metrics import evaluate_generation

predictions = [
    "The cat sat on the mat.",
    "AI is transforming industries."
]

references = [
    "The cat was sitting on the mat.",
    "Artificial intelligence is transforming many industries."
]

metrics = evaluate_generation(predictions, references)

print(f"ROUGE-1: {metrics['rouge']['rouge1']:.4f}")
print(f"ROUGE-2: {metrics['rouge']['rouge2']:.4f}")
print(f"ROUGE-L: {metrics['rouge']['rougeL']:.4f}")
```

## 📚 Detailed Usage

### Classification Metrics

```python
from evaluation_framework.metrics import ClassificationMetrics

cm = ClassificationMetrics()

# Individual metrics
accuracy = cm.accuracy(y_true, y_pred)
metrics = cm.precision_recall_f1(y_true, y_pred, average='weighted')
matrix = cm.confusion_matrix(y_true, y_pred)
report = cm.classification_report(y_true, y_pred, target_names=['pos', 'neg', 'neu'])

# All metrics at once
all_metrics = cm.evaluate_all(y_true, y_pred)
```

#### Averaging Strategies

```python
# Binary classification
metrics = cm.precision_recall_f1(y_true, y_pred, average='binary')

# Macro average (unweighted mean)
metrics = cm.precision_recall_f1(y_true, y_pred, average='macro')

# Weighted average (by support)
metrics = cm.precision_recall_f1(y_true, y_pred, average='weighted')

# Micro average (global)
metrics = cm.precision_recall_f1(y_true, y_pred, average='micro')
```

### QA Metrics

```python
from evaluation_framework.metrics import QAMetrics

# Single pair
em_score = QAMetrics.exact_match("Paris", "Paris, France")
f1_score = QAMetrics.f1_score("Paris", "Paris, France")

# Batch evaluation
predictions = ["answer1", "answer2", "answer3"]
references = ["answer1", "answer2", "different"]

metrics = QAMetrics.evaluate_batch(predictions, references)
print(f"Average EM: {metrics['exact_match']:.4f}")
print(f"Average F1: {metrics['f1']:.4f}")
```

#### Answer Normalization

The QA metrics automatically normalize answers:
- Lowercase
- Remove punctuation
- Remove articles (a, an, the)
- Normalize whitespace

```python
# These are considered matches
QAMetrics.exact_match("The Paris", "paris")  # 1.0
QAMetrics.exact_match("New York City", "new york city")  # 1.0
```

### Generation Metrics

```python
from evaluation_framework.metrics import GenerationMetrics

gm = GenerationMetrics()

# BLEU score (multiple references per prediction)
predictions = ["The cat sat on the mat"]
references = [["The cat sat on the mat", "A cat was sitting on the mat"]]

bleu = gm.bleu_score(predictions, references, max_order=4)
print(f"BLEU-4: {bleu['bleu']:.4f}")

# ROUGE scores (single reference)
predictions = ["The cat sat on the mat"]
references = ["The cat was sitting on the mat"]

rouge = gm.rouge_score(predictions, references)
print(f"ROUGE-1: {rouge['rouge1']:.4f}")
print(f"ROUGE-2: {rouge['rouge2']:.4f}")
print(f"ROUGE-L: {rouge['rougeL']:.4f}")

# METEOR score
meteor = gm.meteor_score(predictions, references)
print(f"METEOR: {meteor:.4f}")
```

### Perplexity

```python
from evaluation_framework.metrics import PerplexityMetric

# From single loss
perplexity = PerplexityMetric.calculate(loss=2.5)
print(f"Perplexity: {perplexity:.2f}")

# From batch of losses
losses = [2.3, 2.5, 2.7, 2.4]
avg_perplexity = PerplexityMetric.calculate_batch(losses)
print(f"Average Perplexity: {avg_perplexity:.2f}")
```

### Unified Evaluator

```python
from evaluation_framework.metrics import ModelEvaluator

# For classification
evaluator = ModelEvaluator(task='classification')
metrics = evaluator.evaluate(predictions, references)
evaluator.print_report(metrics)

# For QA
evaluator = ModelEvaluator(task='qa')
metrics = evaluator.evaluate(predictions, references)
evaluator.print_report(metrics)

# For generation
evaluator = ModelEvaluator(task='generation')
metrics = evaluator.evaluate(predictions, references)
evaluator.print_report(metrics)
```

## 🎨 Real-World Examples

### Sentiment Analysis Evaluation

```python
from evaluation_framework.metrics import evaluate_classification

# Model predictions
predictions = []
ground_truth = []

for text, label in test_data:
    pred = sentiment_model.predict(text)
    predictions.append(pred)
    ground_truth.append(label)

# Evaluate
metrics = evaluate_classification(
    ground_truth,
    predictions,
    target_names=['negative', 'neutral', 'positive']
)

print(metrics['report'])
```

### QA System Evaluation

```python
from evaluation_framework.metrics import evaluate_qa

predictions = []
references = []

for item in qa_test_set:
    pred = qa_model.answer(item['question'], item['context'])
    predictions.append(pred)
    references.append(item['answer'])

metrics = evaluate_qa(predictions, references)
print(f"System EM: {metrics['exact_match']:.2%}")
print(f"System F1: {metrics['f1']:.2%}")
```

### Summarization Evaluation

```python
from evaluation_framework.metrics import evaluate_generation

predictions = []
references = []

for article in articles:
    summary = summarizer.summarize(article['text'])
    predictions.append(summary)
    references.append(article['reference_summary'])

metrics = evaluate_generation(predictions, references)

print("ROUGE Scores:")
for metric, score in metrics['rouge'].items():
    print(f"  {metric}: {score:.4f}")
```

### Multi-Task Evaluation

```python
from evaluation_framework.metrics import ModelEvaluator

results = {}

# Task 1: Classification
eval_cls = ModelEvaluator('classification')
results['classification'] = eval_cls.evaluate(cls_pred, cls_true)

# Task 2: QA
eval_qa = ModelEvaluator('qa')
results['qa'] = eval_qa.evaluate(qa_pred, qa_true)

# Task 3: Generation
eval_gen = ModelEvaluator('generation')
results['generation'] = eval_gen.evaluate(gen_pred, gen_true)

# Print all reports
for task, metrics in results.items():
    print(f"\n{'='*80}")
    print(f"{task.upper()} RESULTS")
    print('='*80)
    ModelEvaluator(task).print_report(metrics)
```

## 📊 Metrics Cheat Sheet

### Classification

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| Accuracy | Correct / Total | [0, 1] | 1.0 |
| Precision | TP / (TP + FP) | [0, 1] | 1.0 |
| Recall | TP / (TP + FN) | [0, 1] | 1.0 |
| F1 | 2 * (P * R) / (P + R) | [0, 1] | 1.0 |

### QA

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| Exact Match | Normalized exact string match | [0, 1] | 1.0 |
| F1 | Token-level F1 score | [0, 1] | 1.0 |

### Generation

| Metric | Description | Range | Best |
|--------|-------------|-------|------|
| BLEU | N-gram precision | [0, 1] | 1.0 |
| ROUGE-1 | Unigram overlap | [0, 1] | 1.0 |
| ROUGE-2 | Bigram overlap | [0, 1] | 1.0 |
| ROUGE-L | Longest common subsequence | [0, 1] | 1.0 |
| METEOR | Harmonic mean with stemming | [0, 1] | 1.0 |

## 🔧 Advanced Features

### Custom Metrics

```python
from evaluation_framework.metrics import ClassificationMetrics

class CustomMetrics(ClassificationMetrics):
    @staticmethod
    def custom_score(y_true, y_pred):
        # Your custom metric
        score = compute_custom_metric(y_true, y_pred)
        return score
```

### Confidence Intervals

```python
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np

def accuracy_with_ci(y_true, y_pred, confidence=0.95):
    """Calculate accuracy with confidence interval."""
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    
    # Wilson score interval
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2/n
    center = (acc + z**2/(2*n)) / denominator
    margin = z * np.sqrt(acc*(1-acc)/n + z**2/(4*n**2)) / denominator
    
    return acc, (center - margin, center + margin)

acc, (lower, upper) = accuracy_with_ci(y_true, y_pred)
print(f"Accuracy: {acc:.4f} [{lower:.4f}, {upper:.4f}]")
```

### Statistical Significance Testing

```python
from scipy import stats

def mcnemar_test(y_true, pred1, pred2):
    """Compare two models using McNemar's test."""
    correct1 = (y_true == pred1)
    correct2 = (y_true == pred2)
    
    # Contingency table
    n01 = sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    n10 = sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    
    # McNemar's test
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, 1)
    
    return p_value

p = mcnemar_test(y_true, model1_pred, model2_pred)
print(f"P-value: {p:.4f}")
if p < 0.05:
    print("Models are significantly different")
```

## 🛠️ Requirements

```
scikit-learn>=1.3.0
numpy>=1.24.0
evaluate>=0.4.0  # Optional, for BLEU, ROUGE, METEOR
```

## 📚 References

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [SQuAD Metrics](https://rajpurkar.github.io/SQuAD-explorer/)
- [ROUGE Paper](https://aclanthology.org/W04-1013/)
- [BLEU Paper](https://aclanthology.org/P02-1040/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More generation metrics (BERTScore, BLEURT)
- Specialized metrics (NER F1, Semantic similarity)
- Visualization tools
- Statistical tests

## 📄 License

MIT License - see repository LICENSE file
