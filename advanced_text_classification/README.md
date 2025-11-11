# Advanced Text Classification

Modern text classification approaches beyond traditional supervised learning.

## 🎯 Features

- **Zero-Shot Classification** - Classify into arbitrary categories without training
- **Multi-Label Classification** - Assign multiple labels to texts
- **Aspect-Based Sentiment** - Analyze sentiment towards specific aspects/topics
- **No Training Required** - Use pre-trained models out of the box

## 🚀 Quick Start

### Zero-Shot Classification

```python
from advanced_text_classification.model import ZeroShotClassifier

# Initialize classifier
classifier = ZeroShotClassifier()

# Classify without any training!
text = "This new smartphone has an amazing camera but terrible battery life."
labels = ["technology", "food", "sports", "politics", "entertainment"]

result = classifier.classify(text, labels)

print(f"Top label: {result['labels'][0]}")
print(f"Confidence: {result['scores'][0]:.4f}")

# Output:
# Top label: technology
# Confidence: 0.9823
```

### Multi-Label Classification

```python
from advanced_text_classification.model import MultiLabelClassifier

# Initialize classifier (requires training/fine-tuned model)
classifier = MultiLabelClassifier(num_labels=10)

# Set label names
classifier.set_label_names([
    "politics", "technology", "sports", "entertainment", "science",
    "health", "business", "education", "environment", "culture"
])

# Predict (can assign multiple labels)
text = "AI revolutionizes healthcare with new diagnostic tools."
result = classifier.predict(text, threshold=0.5)

print(f"Labels: {result['labels']}")
print(f"Scores: {result['scores']}")

# Output:
# Labels: ['technology', 'health', 'science']
# Scores: [0.92, 0.87, 0.76]
```

### Aspect-Based Sentiment

```python
from advanced_text_classification.model import AspectBasedSentiment

# Initialize analyzer
analyzer = AspectBasedSentiment()

# Analyze sentiment for specific aspects
text = "The food was delicious but the service was slow."
aspects = ["food", "service"]

results = analyzer.analyze_multiple_aspects(text, aspects)

for result in results:
    print(f"Aspect: {result['aspect']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Score: {result['score']:.4f}\n")

# Output:
# Aspect: food
# Sentiment: positive
# Score: 0.94
#
# Aspect: service
# Sentiment: negative
# Score: 0.89
```

## 📚 Detailed Usage

### Zero-Shot Classification Options

```python
classifier = ZeroShotClassifier(model_name="facebook/bart-large-mnli")

# Single-label classification
result = classifier.classify(
    text="Python is a great programming language",
    candidate_labels=["technology", "food", "sports"],
    multi_label=False
)

# Multi-label classification
result = classifier.classify(
    text="This article discusses AI and healthcare innovations",
    candidate_labels=["technology", "health", "politics", "sports"],
    multi_label=True  # Can assign multiple labels
)

# Custom hypothesis template
result = classifier.classify(
    text="I love this product!",
    candidate_labels=["positive", "negative"],
    hypothesis_template="The sentiment is {}."
)

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
results = classifier.classify_batch(
    texts,
    candidate_labels=["cat1", "cat2", "cat3"],
    batch_size=4
)
```

### Available Zero-Shot Models

- `facebook/bart-large-mnli` (Default, Recommended) - 400M params, high quality
- `facebook/bart-base-mnli` - Faster, smaller
- `microsoft/deberta-v3-large-mnli` - Very high quality
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` - Multilingual support

### Multi-Label Classification

```python
# Initialize with custom model
classifier = MultiLabelClassifier(
    model_name="bert-base-uncased",
    num_labels=20
)

# Get top K labels regardless of threshold
result = classifier.predict(
    text="Your text here",
    top_k=5  # Get top 5 labels
)

# Use higher threshold for precision
result = classifier.predict(
    text="Your text here",
    threshold=0.8  # Only labels with >0.8 probability
)

# Batch prediction
texts = ["Text 1", "Text 2"]
results = classifier.predict_batch(texts, threshold=0.5, batch_size=8)
```

## 🎨 Use Cases

### 1. Content Categorization

```python
# Categorize articles without training
classifier = ZeroShotClassifier()

article = "Scientists discover new exoplanet in habitable zone..."
categories = [
    "science", "technology", "politics", "business",
    "entertainment", "sports", "health"
]

result = classifier.classify(article, categories)
category = result['labels'][0]
```

### 2. Intent Classification

```python
# Classify user intents in chatbots
user_message = "I want to cancel my subscription"
intents = [
    "cancel_subscription",
    "get_support",
    "check_status",
    "make_payment",
    "ask_question"
]

result = classifier.classify(
    user_message,
    intents,
    hypothesis_template="The user wants to {}."
)
```

### 3. Topic Modeling

```python
# Find topics in documents
document = "Long document text..."
topics = [
    "machine learning", "deep learning", "nlp",
    "computer vision", "robotics", "data science"
]

result = classifier.classify(
    document,
    topics,
    multi_label=True  # Document can have multiple topics
)
```

### 4. Sentiment Analysis by Aspect

```python
# Analyze product reviews
review = "Great camera quality but battery drains too fast."
aspects = ["camera", "battery", "screen", "performance", "price"]

analyzer = AspectBasedSentiment()
results = analyzer.analyze_multiple_aspects(review, aspects)

# Extract insights
for r in results:
    if r['score'] > 0.7:  # High confidence
        print(f"{r['aspect']}: {r['sentiment']}")
```

### 5. Multi-Label Document Tagging

```python
# Auto-tag documents with multiple labels
classifier = MultiLabelClassifier(num_labels=50)
classifier.set_label_names([
    "python", "javascript", "machine-learning", "web-dev",
    # ... 46 more tags
])

document = "Tutorial on building ML models with Python..."
tags = classifier.predict(document, threshold=0.6)

print(f"Auto-generated tags: {tags['labels']}")
```

## 📊 Comparison

| Method | Training | Speed | Flexibility | Best For |
|--------|----------|-------|-------------|----------|
| Zero-Shot | ❌ None | Fast | ⭐⭐⭐⭐⭐ | Ad-hoc categories |
| Multi-Label | ✅ Required | Fast | ⭐⭐⭐ | Fixed label sets |
| Aspect Sentiment | ❌ None | Medium | ⭐⭐⭐⭐ | Detailed reviews |

## 🔧 Advanced Features

### Custom Hypothesis Templates

```python
# For better zero-shot results
classifier.classify(
    "I hate bugs in my code!",
    ["angry", "happy", "frustrated"],
    hypothesis_template="The person is feeling {}."
)

# For classification tasks
classifier.classify(
    "Red sports car for sale",
    ["vehicle", "property", "electronics"],
    hypothesis_template="This is a {}."
)
```

### Handling Ambiguity

```python
# Get confidence for all labels
result = classifier.classify(
    ambiguous_text,
    candidate_labels,
    multi_label=False
)

# Check confidence spread
top_score = result['scores'][0]
second_score = result['scores'][1]

if top_score - second_score < 0.1:
    print("Warning: Low confidence, ambiguous classification")
```

### Threshold Tuning

```python
# For multi-label, tune threshold based on your needs

# High precision (fewer false positives)
result = classifier.predict(text, threshold=0.8)

# High recall (fewer false negatives)
result = classifier.predict(text, threshold=0.3)

# Balanced
result = classifier.predict(text, threshold=0.5)
```

## 📈 Performance Tips

### For Speed
```python
# Use smaller models
classifier = ZeroShotClassifier(model_name="facebook/bart-base-mnli")

# Batch processing
results = classifier.classify_batch(texts, labels, batch_size=16)

# Limit candidate labels
result = classifier.classify(text, most_relevant_labels[:5])
```

### For Quality
```python
# Use larger models
classifier = ZeroShotClassifier(model_name="microsoft/deberta-v3-large-mnli")

# Use custom hypothesis templates
result = classifier.classify(
    text,
    labels,
    hypothesis_template="The main topic is {}."
)

# For multi-label, use appropriate threshold
result = classifier.predict(text, threshold=0.7)
```

## 🛠️ Requirements

```
transformers>=4.30.0
torch>=2.0.0
```

## 📚 References

- [Zero-Shot Learning with BART](https://arxiv.org/abs/1910.13461)
- [Multi-Label Classification](https://arxiv.org/abs/1312.6229)
- [Aspect-Based Sentiment](https://aclanthology.org/S14-2004/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Few-shot learning examples
- More aspect sentiment models
- Fine-tuning guides for multi-label
- Cross-lingual classification

## 📄 License

MIT License - see repository LICENSE file
