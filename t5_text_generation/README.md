# T5 Text Generation Project

This project implements **T5 (Text-to-Text Transfer Transformer)** for various text generation tasks including summarization, translation, paraphrasing, and general text generation.

## 🎯 Features

- **Text Summarization**: Generate concise summaries from long documents
- **Paraphrasing**: Rewrite text while preserving meaning
- **Translation**: Multi-language translation support
- **Question Answering**: Generate answers from context
- **General Text Generation**: Custom text-to-text tasks
- **Fine-tuning Support**: Train on custom datasets
- **Batch Processing**: Efficient batch inference
- **Multiple T5 Variants**: Support for T5-small, T5-base, T5-large

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from t5_text_generation import T5Generator

# Initialize the generator
generator = T5Generator(model_name="t5-base")

# Summarization
summary = generator.summarize(
    "Your long text here...",
    max_length=150,
    min_length=50
)

# Paraphrasing
paraphrase = generator.paraphrase(
    "The quick brown fox jumps over the lazy dog."
)

# Translation
translation = generator.translate(
    "Hello, how are you?",
    source_lang="en",
    target_lang="de"
)

# Question Answering
answer = generator.answer_question(
    question="What is AI?",
    context="Artificial Intelligence is..."
)
```

### Command Line Interface

```bash
# Summarize text
python -m t5_text_generation.inference \
    --task summarize \
    --input "Your text here" \
    --max-length 100

# Paraphrase text
python -m t5_text_generation.inference \
    --task paraphrase \
    --input "Your text here"

# Translate text
python -m t5_text_generation.inference \
    --task translate \
    --input "Hello world" \
    --source-lang en \
    --target-lang fr
```

### Fine-tuning

```python
from t5_text_generation import T5Trainer

# Initialize trainer
trainer = T5Trainer(
    model_name="t5-small",
    output_dir="./models/t5-finetuned"
)

# Prepare your dataset
train_data = [
    {"input": "summarize: Long text...", "output": "Summary..."},
    {"input": "paraphrase: Original text", "output": "Paraphrased text"},
]

# Train the model
trainer.train(
    train_data=train_data,
    epochs=3,
    batch_size=8,
    learning_rate=5e-5
)
```

## 📊 Evaluation

```python
from t5_text_generation import T5Evaluator

evaluator = T5Evaluator(model_name="t5-base")

# Evaluate summarization
metrics = evaluator.evaluate_summarization(
    test_data,
    metrics=["rouge", "bleu"]
)

print(f"ROUGE-1: {metrics['rouge1']}")
print(f"ROUGE-2: {metrics['rouge2']}")
print(f"ROUGE-L: {metrics['rougeL']}")
```

## 🎨 Supported Tasks

| Task | Prefix | Example |
|------|--------|---------|
| Summarization | `summarize:` | `summarize: The article discusses...` |
| Translation | `translate English to German:` | `translate English to German: Hello` |
| Paraphrasing | `paraphrase:` | `paraphrase: The cat sat on the mat` |
| Question Answering | `question: ... context:` | `question: What is AI? context: AI is...` |
| Sentiment | `sentiment:` | `sentiment: This movie is great!` |

## 🔧 Configuration

```python
from t5_text_generation import T5Config

config = T5Config(
    model_name="t5-base",
    max_length=512,
    num_beams=4,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    device="cuda"  # or "cpu"
)

generator = T5Generator(config=config)
```

## 📈 Performance Tips

1. **Use smaller models for faster inference**: T5-small or T5-base
2. **Enable GPU acceleration**: Set `device="cuda"`
3. **Batch processing**: Process multiple texts at once
4. **Optimize beam search**: Reduce `num_beams` for faster generation
5. **Cache models**: Models are automatically cached after first load

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_inference.py -v

# Run with coverage
pytest tests/ --cov=t5_text_generation
```

## 📚 Examples

See the `examples/` directory for:
- `demo.py` - Basic usage examples
- `summarization_demo.py` - Advanced summarization
- `finetuning_demo.py` - Custom model training
- `batch_processing.py` - Efficient batch inference

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- New features include tests

## 📄 License

MIT License - see repository LICENSE file

## 🔗 References

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [Hugging Face T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [T5 Training Guide](https://huggingface.co/blog/t5-fine-tuning)
