# Question Answering System

Comprehensive question answering system supporting extractive, generative, and hybrid approaches.

## 🎯 Features

- **Extractive QA** - Find answer spans in provided context (BERT/RoBERTa-based)
- **Generative QA** - Generate answers from context (T5/FLAN-based)
- **Hybrid QA** - Combine both approaches for best results
- **Batch Processing** - Efficient processing of multiple questions
- **Confidence Scoring** - Get confidence scores for extractive answers
- **Open-domain QA** - Answer questions without provided context (generative)

## 🚀 Quick Start

### Extractive QA

```python
from question_answering.model import ExtractiveQA

# Initialize model
qa = ExtractiveQA(model_name="deepset/roberta-base-squad2")

# Answer question
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889, it was initially criticized by some of France's leading
artists and intellectuals for its design, but it has become a global cultural icon.
"""

question = "When was the Eiffel Tower built?"

answers = qa.answer(question, context, top_k=3)
for answer in answers:
    print(f"Answer: {answer['answer']}")
    print(f"Score: {answer['score']:.4f}")
    print()
```

### Generative QA

```python
from question_answering.model import GenerativeQA

# Initialize model
qa = GenerativeQA(model_name="google/flan-t5-base")

# Answer with context
context = "Python was created by Guido van Rossum and first released in 1991."
question = "Who created Python?"

answer = qa.answer(question, context)
print(answer)  # "Guido van Rossum"

# Open-domain (without context)
question = "What is the capital of France?"
answer = qa.answer(question)
print(answer)  # "Paris"
```

### Hybrid QA

```python
from question_answering.model import HybridQA

# Initialize hybrid system
qa = HybridQA()

# Answer with automatic method selection
result = qa.answer(question, context, mode="auto")

print(f"Answer: {result['answer']}")
print(f"Method: {result['method']}")
if 'score' in result:
    print(f"Confidence: {result['score']:.4f}")
```

## 📚 Available Models

### Extractive QA Models

- `deepset/roberta-base-squad2` (Recommended) - RoBERTa fine-tuned on SQuAD 2.0
- `distilbert-base-cased-distilled-squad` - Fast DistilBERT model
- `deepset/bert-base-cased-squad2` - BERT fine-tuned on SQuAD 2.0
- `allenai/longformer-large-4096-finetuned-triviaqa` - For long contexts

### Generative QA Models

- `google/flan-t5-base` (Recommended) - Balanced performance
- `google/flan-t5-large` - Better quality
- `facebook/bart-large-cnn` - Good for summarization-style QA
- `t5-base` - Standard T5 model

## 📖 Examples

### Batch Processing

```python
questions = [
    "What is AI?",
    "Who invented the telephone?",
    "When was Python created?"
]

contexts = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines...",
    "The telephone was invented by Alexander Graham Bell in 1876.",
    "Python was created by Guido van Rossum in 1991."
]

# Extractive batch
answers = qa.batch_answer(questions, contexts)
for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a[0]['answer']}\n")

# Generative batch
answers = qa.batch_answer(questions, contexts, batch_size=4)
for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

### Confidence Filtering

```python
qa = ExtractiveQA()

# Only return answers above confidence threshold
answer = qa.get_answer_with_confidence(
    question,
    context,
    confidence_threshold=0.7
)

if answer:
    print(f"High-confidence answer: {answer['answer']}")
else:
    print("No confident answer found")
```

### Long Context Handling

```python
# For very long contexts, use Longformer
qa = ExtractiveQA(model_name="allenai/longformer-large-4096-finetuned-triviaqa")

# Can handle contexts up to 4096 tokens
long_context = "..." * 1000
answer = qa.answer(question, long_context)
```

## 🎛️ Advanced Usage

### Custom Generation Parameters

```python
qa = GenerativeQA()

answer = qa.answer(
    question,
    context,
    max_length=100,       # Maximum answer length
    min_length=10,        # Minimum answer length
    num_beams=4,          # Beam search width
    temperature=1.0,      # Sampling temperature
    top_k=50,            # Top-k sampling
    top_p=0.95           # Nucleus sampling
)
```

### Hybrid Mode Selection

```python
qa = HybridQA()

# Force extractive
result = qa.answer(question, context, mode="extractive")

# Force generative
result = qa.answer(question, context, mode="generative")

# Automatic (tries extractive first, falls back to generative)
result = qa.answer(
    question,
    context,
    mode="auto",
    extractive_threshold=0.5  # Confidence threshold
)
```

## 🎨 Use Cases

### 1. Document QA
```python
# Answer questions from documents
document = load_document("annual_report.pdf")
qa = ExtractiveQA()

questions = [
    "What was the revenue in 2023?",
    "Who is the CEO?",
    "What are the main products?"
]

for q in questions:
    answer = qa.answer(q, document)
    print(f"{q}: {answer[0]['answer']}")
```

### 2. Customer Support
```python
# FAQ answering
faq_database = load_faqs()
qa = HybridQA()

customer_question = "How do I reset my password?"
answer = qa.answer(customer_question, faq_database)
```

### 3. Research Assistant
```python
# Answer questions from research papers
paper_text = load_paper("machine_learning.pdf")
qa = GenerativeQA()

questions = [
    "What is the main contribution?",
    "What datasets were used?",
    "What are the limitations?"
]

for q in questions:
    answer = qa.answer(q, paper_text)
    print(f"{q}\n{answer}\n")
```

### 4. Educational Tools
```python
# Quiz generation and answering
textbook_section = "..."
qa = ExtractiveQA()

# Student asks question
student_question = "What is photosynthesis?"
answer = qa.answer(student_question, textbook_section)
```

### 5. Content Analysis
```python
# Extract information from articles
article = "..."
qa = ExtractiveQA()

# Extract key facts
facts = {
    "Who": qa.answer("Who is mentioned?", article)[0]['answer'],
    "What": qa.answer("What happened?", article)[0]['answer'],
    "When": qa.answer("When did it happen?", article)[0]['answer'],
    "Where": qa.answer("Where did it happen?", article)[0]['answer']
}
```

## 📊 Evaluation

### Metrics

```python
from evaluation_framework.metrics import QAMetrics

predictions = ["Paris", "1991", "Alexander Graham Bell"]
references = ["Paris", "1991", "Alexander Graham Bell"]

metrics = QAMetrics.evaluate_batch(predictions, references)
print(f"Exact Match: {metrics['exact_match']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

### Custom Evaluation

```python
# Evaluate on your dataset
test_data = load_test_data()

predictions = []
for item in test_data:
    answer = qa.answer(item['question'], item['context'])
    predictions.append(answer[0]['answer'] if isinstance(answer, list) else answer)

references = [item['answer'] for item in test_data]

metrics = QAMetrics.evaluate_batch(predictions, references)
```

## 🔍 Model Comparison

| Model | Type | Speed | Quality | Max Context | Best For |
|-------|------|-------|---------|-------------|----------|
| RoBERTa-SQuAD | Extractive | Fast | High | 512 | Factual QA |
| DistilBERT-SQuAD | Extractive | Very Fast | Good | 512 | Speed-critical |
| Longformer | Extractive | Slow | High | 4096 | Long documents |
| FLAN-T5-base | Generative | Medium | High | 512 | Flexible QA |
| FLAN-T5-large | Generative | Slow | Very High | 512 | Best quality |

## 📈 Performance Tips

### For Speed
```python
# Use DistilBERT
qa = ExtractiveQA(model_name="distilbert-base-cased-distilled-squad")

# Reduce top_k
answers = qa.answer(question, context, top_k=1)

# Use GPU
qa = ExtractiveQA(device="cuda")
```

### For Quality
```python
# Use larger model
qa = GenerativeQA(model_name="google/flan-t5-large")

# Increase beam search
answer = qa.answer(question, context, num_beams=8)

# Use hybrid approach
qa = HybridQA()
```

### For Long Documents
```python
# Use Longformer
qa = ExtractiveQA(model_name="allenai/longformer-large-4096-finetuned-triviaqa")

# Or chunk documents for other models
chunks = chunk_document(long_doc, chunk_size=500)
answers = [qa.answer(question, chunk) for chunk in chunks]
best_answer = max(answers, key=lambda x: x[0]['score'])
```

## 🛠️ Requirements

```
transformers>=4.30.0
torch>=2.0.0
```

## 📚 References

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [HuggingFace QA](https://huggingface.co/docs/transformers/task_summary#question-answering)
- [BERT for QA Paper](https://arxiv.org/abs/1810.04805)
- [T5 Paper](https://arxiv.org/abs/1910.10683)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-hop reasoning
- Conversational QA
- Visual QA
- Cross-lingual QA

## 📄 License

MIT License - see repository LICENSE file
