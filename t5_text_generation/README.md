# T5 Text Generation

Comprehensive text-to-text generation using T5 (Text-to-Text Transfer Transformer) models.

## 🎯 Features

- **Text Summarization** - Generate concise summaries of long texts
- **Translation** - Translate between languages
- **Paraphrasing** - Generate alternative phrasings
- **Question Generation** - Create questions from context
- **Grammar Correction** - Fix grammatical errors
- **Batch Processing** - Efficient processing of multiple texts

## 🚀 Quick Start

### Basic Usage

```python
from t5_text_generation.model import T5TextGenerator

# Initialize model
model = T5TextGenerator(model_name="t5-base")

# Summarize text
text = "Your long text here..."
summary = model.summarize(text, max_length=150)
print(summary)

# Translate
translation = model.translate(
    "Hello world",
    source_lang="English",
    target_lang="German"
)
print(translation)

# Paraphrase
paraphrases = model.paraphrase("This is an example sentence.", num_return_sequences=3)
for p in paraphrases:
    print(p)
```

### Available Models

- `t5-small` (60M params) - Fast, good for testing
- `t5-base` (220M params) - Balanced performance
- `t5-large` (770M params) - Best quality
- `t5-3b` / `t5-11b` - Highest quality (requires more memory)

## 📚 Examples

### Summarization

```python
model = T5TextGenerator(model_name="t5-base")

article = """
Artificial intelligence (AI) is intelligence demonstrated by machines,
in contrast to natural intelligence displayed by humans and animals.
Leading AI textbooks define the field as the study of "intelligent agents":
any device that perceives its environment and takes actions that maximize
its chance of successfully achieving its goals.
"""

summary = model.summarize(
    article,
    max_length=60,
    min_length=20,
    num_beams=4
)
print(summary)
# Output: AI is intelligence demonstrated by machines that study intelligent agents.
```

### Translation

```python
# English to German
translation = model.translate(
    "The weather is beautiful today.",
    source_lang="English",
    target_lang="German"
)
print(translation)
# Output: Das Wetter ist heute schön.

# French to English
translation = model.translate(
    "Bonjour, comment allez-vous?",
    source_lang="French",
    target_lang="English"
)
print(translation)
# Output: Hello, how are you?
```

### Question Generation

```python
context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
answer = "Paris"

question = model.generate_question(context, answer)
print(question)
# Output: Where is the Eiffel Tower located?
```

### Batch Processing

```python
texts = [
    "First document to summarize...",
    "Second document to summarize...",
    "Third document to summarize..."
]

summaries = model.batch_generate(
    texts,
    task_prefix="summarize: ",
    max_length=100,
    batch_size=4
)

for text, summary in zip(texts, summaries):
    print(f"Original: {text}")
    print(f"Summary: {summary}\n")
```

## 🎛️ Advanced Parameters

### Generation Parameters

```python
output = model.generate(
    text="Your input text",
    task_prefix="summarize: ",  # Task-specific prefix
    max_length=512,              # Maximum output length
    min_length=10,               # Minimum output length
    num_beams=4,                 # Beam search width
    temperature=1.0,             # Sampling temperature (higher = more random)
    top_k=50,                    # Top-k sampling
    top_p=0.95,                  # Nucleus sampling
    repetition_penalty=1.0,      # Penalty for repetition
    length_penalty=1.0,          # Length penalty for beam search
    no_repeat_ngram_size=3,      # Prevent n-gram repetition
    early_stopping=True          # Stop when all beams finish
)
```

### Temperature Effects

- **Low (0.7)**: More conservative, focused outputs
- **Medium (1.0)**: Balanced randomness
- **High (1.5+)**: More creative, diverse outputs

## 🔧 Training Custom Models

### Prepare Data

```python
from t5_text_generation.train import T5Trainer

# Your training data
train_inputs = [
    "summarize: Your input text 1...",
    "summarize: Your input text 2...",
]
train_targets = [
    "Summary 1",
    "Summary 2",
]

# Initialize trainer
trainer = T5Trainer(model_name="t5-base")

# Prepare datasets
train_dataset, val_dataset = trainer.prepare_data(
    train_inputs,
    train_targets,
    val_inputs=val_inputs,
    val_targets=val_targets
)
```

### Train Model

```python
trainer.train(
    train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    batch_size=8,
    learning_rate=3e-5,
    output_dir="./my_t5_model",
    save_steps=1000,
    eval_steps=500
)
```

### Load Fine-tuned Model

```python
model = T5TextGenerator(model_name="./my_t5_model/best_model")
```

## 🎨 Use Cases

### 1. Document Summarization
- News articles
- Research papers
- Meeting transcripts
- Long emails

### 2. Content Paraphrasing
- SEO content generation
- Text rewriting
- Style transfer
- Content variation

### 3. Question Generation
- Educational content
- Quiz generation
- FAQ creation
- Dataset augmentation

### 4. Grammar Correction
- Writing assistance
- Content editing
- ESL applications
- Quality assurance

### 5. Translation
- Multi-language support
- Content localization
- Cross-lingual understanding

## 📊 Performance Tips

### Memory Optimization

```python
# Use smaller model for faster inference
model = T5TextGenerator(model_name="t5-small")

# Reduce batch size
summaries = model.batch_generate(texts, batch_size=2)

# Limit max length
summary = model.summarize(text, max_length=50)
```

### Speed Optimization

```python
# Use fewer beams
output = model.generate(text, num_beams=2)

# Disable sampling for faster generation
output = model.generate(text, do_sample=False, num_beams=1)
```

### Quality Optimization

```python
# Use larger model
model = T5TextGenerator(model_name="t5-large")

# Increase beam search
output = model.generate(text, num_beams=8)

# Fine-tune on domain-specific data
trainer = T5Trainer(model_name="t5-base")
trainer.train(domain_dataset, epochs=5)
```

## 🧪 Running Examples

```bash
# Run all examples
python -m t5_text_generation.examples.demo

# Individual examples in Python
from t5_text_generation.examples.demo import summarization_example
summarization_example()
```

## 📖 Task Prefixes

T5 uses task prefixes to specify the operation:

- `"summarize: "` - Summarization
- `"translate English to German: "` - Translation
- `"paraphrase: "` - Paraphrasing
- `"grammar: "` - Grammar correction
- `"generate question: "` - Question generation
- Custom prefixes for fine-tuned models

## 🔍 Evaluation

### ROUGE Scores (Summarization)

```python
from evaluate import load

rouge = load("rouge")

predictions = [model.summarize(text) for text in test_texts]
references = [target for target in test_targets]

scores = rouge.compute(predictions=predictions, references=references)
print(scores)
```

### BLEU Scores (Translation)

```python
from evaluate import load

bleu = load("bleu")

predictions = [model.translate(text, "English", "German") for text in test_texts]
references = [[target] for target in test_targets]

scores = bleu.compute(predictions=predictions, references=references)
print(scores)
```

## 🛠️ Requirements

```
transformers>=4.30.0
torch>=2.0.0
```

## 📚 References

- [T5 Paper](https://arxiv.org/abs/1910.10683) - "Exploring the Limits of Transfer Learning"
- [HuggingFace T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [T5 Model Hub](https://huggingface.co/models?search=t5)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional task-specific examples
- More evaluation metrics
- Performance benchmarks
- Domain-specific fine-tuning guides

## 📄 License

MIT License - see repository LICENSE file
