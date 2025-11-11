# GPT-2 Text Generation

Creative text generation and fine-tuning using GPT-2 (Generative Pre-trained Transformer 2).

## 🎯 Features

- **Creative Text Generation** - Generate stories, articles, and creative content
- **Text Completion** - Complete partial texts naturally
- **Dialogue Generation** - Create conversational responses
- **Fine-tuning** - Train on custom datasets for domain-specific generation
- **Controlled Generation** - Control output with temperature and sampling parameters
- **Interactive Mode** - Real-time interactive text generation

## 🚀 Quick Start

### Basic Usage

```python
from gpt2_text_generation.model import GPT2Generator

# Initialize model
generator = GPT2Generator(model_name="gpt2")

# Generate text
prompt = "Once upon a time in a distant land,"
generated = generator.generate(prompt, max_length=100)[0]
print(generated)

# Complete text
partial = "The future of artificial intelligence is"
completed = generator.complete_text(partial, max_new_tokens=50)
print(completed)
```

### Available Models

- `gpt2` (124M params) - Fast, good for testing
- `gpt2-medium` (355M params) - Better quality
- `gpt2-large` (774M params) - High quality
- `gpt2-xl` (1.5B params) - Best quality (requires more memory)

## 📚 Examples

### Story Generation

```python
generator = GPT2Generator(model_name="gpt2")

prompt = "In the year 2050, technology had advanced beyond imagination."
story = generator.generate_story(
    prompt,
    max_length=300,
    temperature=0.9  # Higher temperature for creativity
)
print(story)
```

### Dialogue Generation

```python
context = "Person A: How's the weather today?\nPerson B:"
dialogue = generator.generate_dialogue(
    context,
    num_turns=3,
    max_turn_length=50
)

for i, turn in enumerate(dialogue, 1):
    print(f"Turn {i}: {turn}")
```

### Generate Variations

```python
prompt = "The key to success is"
variations = generator.generate_variations(
    prompt,
    num_variations=5,
    temperature=1.2
)

for i, var in enumerate(variations, 1):
    print(f"{i}. {var}\n")
```

### Controlled Generation

```python
# Conservative (focused)
output = generator.generate(
    "AI will change the world by",
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)[0]

# Creative (diverse)
output = generator.generate(
    "AI will change the world by",
    temperature=1.5,
    top_k=50,
    top_p=0.95
)[0]
```

## 🎛️ Generation Parameters

### Temperature
- **Low (0.5-0.7)**: More focused, deterministic outputs
- **Medium (0.8-1.0)**: Balanced creativity
- **High (1.2-2.0)**: More creative, diverse outputs

### Top-k Sampling
- Limits vocabulary to top k tokens
- Lower k = more focused (try k=40)
- Higher k = more diverse (try k=100)

### Top-p (Nucleus) Sampling
- Samples from smallest set with cumulative probability >= p
- p=0.9: balanced
- p=0.95: more diverse

### Repetition Penalty
- Penalizes repeated tokens
- 1.0 = no penalty
- 1.2-1.5 = good for reducing repetition

## 🔧 Fine-tuning

### Prepare Data

```python
from gpt2_text_generation.train import GPT2Trainer, load_text_file

# Load your training data
train_texts = load_text_file("your_data.txt")

# Or create text list
train_texts = [
    "Your training text 1...",
    "Your training text 2...",
    # ...
]

# Initialize trainer
trainer = GPT2Trainer(model_name="gpt2")

# Prepare datasets
train_dataset, val_dataset, data_collator = trainer.prepare_data(
    train_texts,
    val_texts=val_texts  # Optional
)
```

### Train Model

```python
trainer.train(
    train_dataset,
    data_collator,
    val_dataset=val_dataset,
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    output_dir="./my_gpt2_model",
    save_steps=1000,
    eval_steps=500
)
```

### Use Fine-tuned Model

```python
generator = GPT2Generator(model_name="./my_gpt2_model/best_model")
text = generator.generate("Your domain-specific prompt...")[0]
```

## 🎨 Use Cases

### 1. Creative Writing
- Story generation
- Poetry creation
- Character dialogue
- Plot development

### 2. Content Creation
- Blog post ideas
- Article outlines
- Marketing copy
- Product descriptions

### 3. Code Generation
```python
prompt = "def fibonacci(n):\n    "
code = generator.complete_text(prompt, max_new_tokens=100)
```

### 4. Conversational AI
- Chatbot responses
- Customer service
- Virtual assistants
- Interactive storytelling

### 5. Data Augmentation
- Generate training examples
- Paraphrase existing text
- Create variations

## 📊 Best Practices

### For Creative Writing
```python
generator.generate(
    prompt,
    temperature=1.2,      # High creativity
    top_p=0.95,
    repetition_penalty=1.3,
    max_length=500
)
```

### For Factual/Technical Content
```python
generator.generate(
    prompt,
    temperature=0.7,      # More focused
    top_p=0.9,
    repetition_penalty=1.1,
    max_length=300
)
```

### For Code Generation
```python
generator.generate(
    code_prompt,
    temperature=0.5,      # Very focused
    top_k=40,
    repetition_penalty=1.0,
    max_length=200
)
```

## 🧪 Interactive Mode

```python
generator = GPT2Generator()
generator.interactive_generation(
    initial_prompt="",
    max_iterations=10
)
```

Commands in interactive mode:
- Type prompt and press Enter to generate
- Type `continue` to extend last generation
- Type `quit` or `exit` to stop

## 📈 Performance Tips

### Memory Optimization
```python
# Use smaller model
generator = GPT2Generator(model_name="gpt2")

# Limit max_length
output = generator.generate(prompt, max_length=100)

# Use greedy decoding (faster)
output = generator.generate(prompt, do_sample=False)
```

### Speed Optimization
```python
# Reduce beam search (if using)
output = generator.generate(prompt, num_beams=1)

# Use GPU if available
generator = GPT2Generator(device="cuda")
```

### Quality Optimization
```python
# Use larger model
generator = GPT2Generator(model_name="gpt2-large")

# Increase sampling diversity
output = generator.generate(
    prompt,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)
```

## 🔍 Common Issues

### Repetitive Output
- Increase `repetition_penalty` to 1.2-1.5
- Increase `no_repeat_ngram_size` to 3-5
- Lower `temperature` slightly

### Incoherent Output
- Lower `temperature` to 0.7-0.9
- Reduce `top_k` to 40
- Use beam search with `num_beams=4`

### Too Conservative
- Increase `temperature` to 1.2+
- Increase `top_p` to 0.98
- Increase `top_k` to 100

## 🛠️ Requirements

```
transformers>=4.30.0
torch>=2.0.0
tqdm>=4.65.0
```

## 📚 References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPT-2 Blog Post](https://openai.com/blog/better-language-models/)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More fine-tuning examples
- Domain-specific models
- Advanced generation techniques
- Performance benchmarks

## 📄 License

MIT License - see repository LICENSE file
