# Question Answering System

Advanced extractive and generative question answering using transformer models.

## 🎯 Features

- **Extractive QA**: Answer questions by extracting spans from context
- **Generative QA**: Generate answers using seq2seq models
- **SQuAD Support**: Train and evaluate on SQuAD dataset
- **Multiple Models**: BERT, RoBERTa, DistilBERT support
- **Batch Processing**: Efficient batch question answering
- **Confidence Scores**: Answer confidence estimation
- **Context Retrieval**: Integrate with document retrieval systems

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Question Answering

```python
from question_answering import QASystem

# Initialize QA system
qa = QASystem(model_name="distilbert-base-cased-distilled-squad")

# Ask a question
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris,
France. It is named after the engineer Gustave Eiffel, whose company designed
and built the tower. Constructed from 1887 to 1889, it was initially criticized
by some of France's leading artists and intellectuals for its design.
"""

question = "Who designed the Eiffel Tower?"
answer = qa.answer(question, context)

print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['score']:.2f}")
```

### Batch Processing

```python
# Multiple questions for same context
questions = [
    "Where is the Eiffel Tower located?",
    "When was it constructed?",
    "Who was it named after?"
]

answers = qa.batch_answer(questions, context)
for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a['answer']} (score: {a['score']:.2f})\n")
```

### Command Line Interface

```bash
# Answer a question
python -m question_answering.inference \
    --question "What is AI?" \
    --context "Artificial Intelligence is the simulation of human intelligence..."

# Interactive mode
python -m question_answering.inference --interactive
```

## 🔧 Training on SQuAD

```python
from question_answering import QATrainer

# Initialize trainer
trainer = QATrainer(
    model_name="bert-base-uncased",
    output_dir="./models/qa-bert"
)

# Train on SQuAD
trainer.train_on_squad(
    dataset_version="squad_v2",
    epochs=3,
    batch_size=16
)
```

## 📊 Evaluation

```python
from question_answering import QAEvaluator

evaluator = QAEvaluator(model_name="models/qa-bert")

# Evaluate on test set
metrics = evaluator.evaluate_squad(
    dataset="squad_v2",
    split="validation"
)

print(f"Exact Match: {metrics['exact_match']:.2f}%")
print(f"F1 Score: {metrics['f1']:.2f}")
```

## 🎨 Advanced Features

### Multi-Document QA

```python
# Answer from multiple documents
documents = [doc1, doc2, doc3]
answer = qa.answer_from_documents(question, documents)
```

### Confidence Filtering

```python
# Only return high-confidence answers
answer = qa.answer(
    question,
    context,
    min_confidence=0.7
)
```

### Top-K Answers

```python
# Get multiple possible answers
answers = qa.answer(
    question,
    context,
    top_k=3
)
```

## 📈 Supported Models

- `bert-base-uncased`
- `distilbert-base-cased-distilled-squad`
- `roberta-base-squad2`
- `deepset/bert-base-cased-squad2`
- Custom fine-tuned models

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📚 Examples

- `demo.py` - Basic usage
- `squad_evaluation.py` - SQuAD evaluation
- `document_qa.py` - Multi-document QA
- `training_demo.py` - Model training

## 🤝 Contributing

Contributions welcome! Ensure code quality and test coverage.

## 📄 License

MIT License

## 🔗 References

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [BERT for QA](https://huggingface.co/docs/transformers/tasks/question_answering)
