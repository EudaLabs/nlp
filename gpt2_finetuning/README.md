# GPT-2 Fine-tuning Project

Fine-tune GPT-2 models for text generation, completion, and creative writing tasks.

## 🎯 Features

- **Text Generation**: Generate creative text from prompts
- **Text Completion**: Complete partial sentences and paragraphs
- **Fine-tuning**: Train on custom datasets
- **Multiple Model Sizes**: Support for GPT-2 small, medium, large, and XL
- **Controlled Generation**: Temperature, top-k, top-p sampling
- **Batch Processing**: Efficient batch text generation
- **Prompt Engineering**: Built-in prompt templates

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Text Generation

```python
from gpt2_finetuning import GPT2Generator

# Initialize generator
generator = GPT2Generator(model_name="gpt2")

# Generate text
text = generator.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8
)
print(text)
```

### Text Completion

```python
# Complete a sentence
completion = generator.complete(
    prompt="The future of artificial intelligence is",
    max_length=50
)
print(completion)
```

### Creative Writing

```python
# Generate creative content
story = generator.generate(
    prompt="In a world where robots and humans coexist,",
    max_length=200,
    temperature=0.9,
    top_p=0.95
)
```

### Command Line Interface

```bash
# Generate text
python -m gpt2_finetuning.inference \
    --prompt "Once upon a time" \
    --max-length 100 \
    --temperature 0.8

# Complete text
python -m gpt2_finetuning.inference \
    --prompt "The meaning of life is" \
    --max-length 50 \
    --mode complete
```

## 🔧 Fine-tuning

### Prepare Your Dataset

```python
# Create training data (plain text)
texts = [
    "Your first training text...",
    "Your second training text...",
    "More training examples..."
]

# Save to file
with open("train.txt", "w") as f:
    for text in texts:
        f.write(text + "\n\n")
```

### Train the Model

```python
from gpt2_finetuning import GPT2Trainer

# Initialize trainer
trainer = GPT2Trainer(
    model_name="gpt2",
    output_dir="./models/gpt2-custom"
)

# Train
trainer.train(
    train_file="train.txt",
    epochs=3,
    batch_size=4,
    learning_rate=5e-5
)
```

### Command Line Training

```bash
python -m gpt2_finetuning.train \
    --train-file train.txt \
    --model-name gpt2 \
    --output-dir ./models/gpt2-custom \
    --epochs 3 \
    --batch-size 4
```

## 🎨 Generation Strategies

### Temperature Control

```python
# Low temperature (more focused, deterministic)
focused = generator.generate(prompt, temperature=0.5)

# High temperature (more creative, random)
creative = generator.generate(prompt, temperature=1.2)
```

### Top-k Sampling

```python
# Limit to top 50 tokens
text = generator.generate(prompt, top_k=50)
```

### Top-p (Nucleus) Sampling

```python
# Use nucleus sampling
text = generator.generate(prompt, top_p=0.9)
```

### Beam Search

```python
# Use beam search for more coherent text
text = generator.generate(prompt, num_beams=5)
```

## 📊 Model Variants

| Model | Parameters | Memory | Speed | Quality |
|-------|-----------|--------|-------|---------|
| gpt2 (small) | 124M | ~500MB | Fast | Good |
| gpt2-medium | 355M | ~1.4GB | Medium | Better |
| gpt2-large | 774M | ~3GB | Slow | Great |
| gpt2-xl | 1.5B | ~6GB | Very Slow | Best |

## 🔍 Advanced Features

### Conditional Generation

```python
# Generate with multiple conditions
from gpt2_finetuning import ConditionalGenerator

gen = ConditionalGenerator()
text = gen.generate_conditional(
    prompt="Write a story about",
    conditions={
        "genre": "science fiction",
        "tone": "optimistic",
        "length": "short"
    }
)
```

### Batch Generation

```python
prompts = [
    "Once upon a time",
    "In a galaxy far away",
    "The secret to success is"
]

results = generator.batch_generate(
    prompts,
    max_length=50,
    batch_size=2
)
```

### Prompt Templates

```python
from gpt2_finetuning.utils import PromptTemplate

# Story template
template = PromptTemplate.story(
    setting="medieval castle",
    character="brave knight",
    conflict="dragon attack"
)

story = generator.generate(template.render(), max_length=200)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_inference.py -v

# Run with coverage
pytest tests/ --cov=gpt2_finetuning
```

## 📈 Performance Tips

1. **Start with gpt2 (small)** for faster iteration
2. **Use GPU** for significant speedup
3. **Batch prompts** for efficiency
4. **Cache models** to avoid reloading
5. **Tune generation parameters** for your use case

## 💡 Use Cases

- **Creative Writing**: Stories, poems, scripts
- **Content Generation**: Blog posts, articles, social media
- **Code Generation**: Programming examples
- **Chatbots**: Conversational AI
- **Text Completion**: Auto-complete features
- **Data Augmentation**: Generate training data

## 📚 Examples

See the `examples/` directory for:
- `demo.py` - Basic usage examples
- `creative_writing.py` - Story and poem generation
- `chatbot.py` - Simple conversational AI
- `finetuning_demo.py` - Custom model training

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- New features include tests

## 📄 License

MIT License - see repository LICENSE file

## 🔗 References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPT-2 Training Guide](https://huggingface.co/blog/how-to-generate)
