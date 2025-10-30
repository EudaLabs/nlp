# BERT Text Classification

Fine-tuning BERT for text classification tasks using Hugging Face Transformers.

## Overview

This project demonstrates how to fine-tune a pre-trained BERT model for various text classification tasks including:
- Binary classification (e.g., sentiment analysis, spam detection)
- Multi-class classification (e.g., topic classification, intent detection)

## Features

- ✅ Pre-trained BERT model fine-tuning
- ✅ Support for binary and multi-class classification
- ✅ Training with automatic mixed precision
- ✅ Evaluation metrics (accuracy, F1, precision, recall)
- ✅ Model checkpointing and early stopping
- ✅ Easy-to-use inference API
- ✅ Example with IMDB sentiment analysis dataset

## Installation

```bash
pip install transformers torch datasets scikit-learn numpy pandas tqdm
```

Or install all project dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```python
from bert_classification.train import train_classifier

# Train on IMDB dataset (binary classification)
model, tokenizer, results = train_classifier(
    dataset_name="imdb",
    model_name="bert-base-uncased",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5
)
```

### Inference

```python
from bert_classification.inference import BERTClassifier

# Load trained model
classifier = BERTClassifier("./models/bert-imdb")

# Predict on new text
text = "This movie was absolutely fantastic! I loved every minute of it."
prediction = classifier.predict(text)
print(f"Sentiment: {prediction['label']}, Confidence: {prediction['confidence']:.2f}")
```

### Using the CLI

Train a model:
```bash
python -m bert_classification.train \
    --dataset imdb \
    --model bert-base-uncased \
    --epochs 3 \
    --batch-size 16 \
    --output-dir ./models/bert-imdb
```

Run inference:
```bash
python -m bert_classification.inference \
    --model-path ./models/bert-imdb \
    --text "This movie is amazing!"
```

## Project Structure

```
bert_classification/
├── __init__.py           # Package initialization
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── train.py             # Training script
├── inference.py         # Inference and prediction
├── utils.py             # Utility functions
├── config.py            # Configuration classes
└── examples/
    └── demo.py          # Usage examples
```

## Model Performance

### IMDB Sentiment Analysis (Binary Classification)

| Metric | Score |
|--------|-------|
| Accuracy | 92.5% |
| F1 Score | 92.3% |
| Precision | 92.8% |
| Recall | 91.8% |

*Results after 3 epochs of fine-tuning on IMDB dataset*

## Advanced Usage

### Custom Dataset

```python
from bert_classification.train import train_classifier

# Train on your own dataset
model, tokenizer, results = train_classifier(
    texts=train_texts,
    labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    num_classes=5,  # Multi-class
    model_name="bert-base-uncased",
    epochs=5,
    batch_size=32
)
```

### Multi-class Classification

```python
# Topic classification example
topics = ["sports", "technology", "politics", "entertainment", "business"]

classifier = BERTClassifier("./models/bert-topics", num_classes=len(topics))
text = "Apple announces new iPhone with revolutionary features"
prediction = classifier.predict(text)
print(f"Topic: {topics[prediction['label']]}")
```

## Configuration Options

Key parameters you can configure:

- `model_name`: Pre-trained model (default: "bert-base-uncased")
- `epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size for training (default: 16)
- `learning_rate`: Learning rate (default: 2e-5)
- `max_length`: Maximum sequence length (default: 512)
- `warmup_steps`: Number of warmup steps (default: 500)
- `weight_decay`: Weight decay for optimization (default: 0.01)

## Tips for Best Results

1. **Start with a smaller model** for quick experimentation (distilbert-base-uncased)
2. **Use appropriate batch size** based on your GPU memory
3. **Monitor validation loss** to detect overfitting
4. **Experiment with learning rates** (typical range: 1e-5 to 5e-5)
5. **Consider domain-specific BERT** models for specialized tasks

## Troubleshooting

**Out of Memory Error:**
- Reduce batch size
- Use gradient accumulation
- Try a smaller model (DistilBERT)

**Poor Performance:**
- Train for more epochs
- Check data quality and balance
- Try different learning rates
- Use a domain-specific pre-trained model

## References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [IMDB Dataset](https://huggingface.co/datasets/imdb)

## License

This project is part of the EudaLabs NLP repository and follows the same license.

## Contributing

Contributions are welcome! Please check the main repository's contributing guidelines.
