# 🚀 Quick Start Guide

Get started with the NLP repository in minutes!

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

## ⚡ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/EudaLabs/nlp.git
cd nlp
```

### 2. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Download SpaCy models (if using SpaCy projects)
python -m spacy download en_core_web_sm
```

### 3. Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v
```

## 🎯 First Steps

### Try Your First Demo

Launch the interactive demo hub to explore all capabilities:

```bash
python -m gradio_demos.demo_hub
```

Then open your browser to `http://localhost:7860`

### Run Individual Demos

```bash
# T5 Text Generation (summarization, translation, paraphrasing)
python -m gradio_demos.t5_generation

# GPT-2 Creative Writing
python -m gradio_demos.gpt2_generation

# Question Answering
python -m gradio_demos.question_answering_demo
```

## 📚 Example Usage

### 1. Text Summarization

```python
from t5_text_generation.model import T5TextGenerator

# Initialize model
model = T5TextGenerator(model_name="t5-small")

# Summarize text
text = """
Your long text here...
"""

summary = model.summarize(text, max_length=100)
print(summary)
```

### 2. Question Answering

```python
from question_answering.model import ExtractiveQA

# Initialize QA model
qa = ExtractiveQA()

# Ask a question
context = "The Eiffel Tower is located in Paris, France."
question = "Where is the Eiffel Tower?"

answer = qa.answer(question, context)
print(answer[0]['answer'])  # "Paris, France"
```

### 3. Zero-Shot Classification

```python
from advanced_text_classification.model import ZeroShotClassifier

# Initialize classifier
classifier = ZeroShotClassifier()

# Classify without training!
text = "This movie was absolutely fantastic!"
labels = ["positive", "negative", "neutral"]

result = classifier.classify(text, labels)
print(f"Label: {result['labels'][0]}")  # "positive"
```

### 4. Creative Text Generation

```python
from gpt2_text_generation.model import GPT2Generator

# Initialize generator
generator = GPT2Generator(model_name="gpt2")

# Generate text
prompt = "Once upon a time in a magical forest,"
story = generator.generate_story(prompt, max_length=200)
print(story)
```

## 🗂️ Repository Structure

```
nlp/
├── t5_text_generation/          # T5 text generation
├── gpt2_text_generation/        # GPT-2 text generation
├── question_answering/          # QA systems
├── advanced_text_classification/# Zero-shot, multi-label
├── evaluation_framework/        # Model evaluation
├── bert_classification/         # BERT fine-tuning
├── ner_system/                  # Named entity recognition
├── fastapi_deployment/          # API deployment
├── gradio_demos/               # Interactive demos
├── basic_text_processing/      # Basic NLP
├── word_embeddings/            # Word2Vec, etc.
├── learning_spacy/             # SpaCy projects
├── langchain/                  # LangChain apps
├── agentic_ai/                 # RAG systems
└── tests/                      # Test suite
```

## 🎓 Learning Path

### Beginner (Start Here)

1. **Text Classification**
   - `advanced_text_classification/` - Zero-shot classification
   - `bert_classification/` - Fine-tuning BERT

2. **Basic NLP**
   - `basic_text_processing/` - Fundamentals
   - `word_embeddings/` - Word2Vec

3. **Interactive Demos**
   - Launch `demo_hub.py` to explore

### Intermediate

1. **Text Generation**
   - `t5_text_generation/` - Summarization, translation
   - `gpt2_text_generation/` - Creative writing

2. **Question Answering**
   - `question_answering/` - Extractive and generative QA

3. **Named Entity Recognition**
   - `ner_system/` - Entity extraction

### Advanced

1. **Model Deployment**
   - `fastapi_deployment/` - REST APIs
   - `gradio_demos/` - Web interfaces

2. **RAG Systems**
   - `agentic_ai/` - Retrieval-augmented generation
   - `langchain/` - LLM chains

3. **Custom Training**
   - Fine-tune models on your data
   - Evaluate with `evaluation_framework/`

## 🔧 Common Tasks

### Train a Model

```bash
# Fine-tune BERT for classification
cd bert_classification
python -m bert_classification.train \
    --dataset your_dataset \
    --epochs 3 \
    --batch-size 16
```

### Deploy a Model

```bash
# Start FastAPI server
cd fastapi_deployment
python -m fastapi_deployment.app
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bert_classification.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Evaluate a Model

```python
from evaluation_framework.metrics import evaluate_classification

y_true = ["positive", "negative", "neutral"]
y_pred = ["positive", "negative", "positive"]

metrics = evaluate_classification(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## 🐛 Troubleshooting

### Model Download Issues

If models fail to download:

```python
# Set HuggingFace cache directory
import os
os.environ['HF_HOME'] = '/path/to/cache'
```

### CUDA/GPU Issues

To use GPU:

```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Force CPU usage
model = Model(device="cpu")
```

### Memory Issues

For large models:

```python
# Use smaller model variants
model = T5TextGenerator(model_name="t5-small")  # Instead of t5-large

# Reduce batch size
results = model.batch_generate(texts, batch_size=2)
```

## 📖 Next Steps

1. **Explore Documentation**
   - Each project has a detailed README
   - Check `ROADMAP.md` for future plans

2. **Try Examples**
   - Run example scripts in each project
   - Modify parameters and experiment

3. **Build Something**
   - Combine multiple modules
   - Create your own application

4. **Contribute**
   - See `CONTRIBUTING.md`
   - Submit PRs with improvements

## 💡 Tips

1. **Start Simple**: Begin with demo hub and examples
2. **Read READMEs**: Each module has comprehensive docs
3. **Use Smaller Models**: Faster for experimentation
4. **Check Examples**: Every project has working examples
5. **Ask Questions**: Open issues for help

## 🔗 Resources

- **Documentation**: Check individual project READMEs
- **Examples**: Look in `examples/` directories
- **Demos**: Launch interactive Gradio apps
- **Tests**: See `tests/` for usage patterns

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/EudaLabs/nlp/issues)
- **Documentation**: Project READMEs
- **Examples**: Working code in `examples/` folders

---

**Ready to dive deeper?** Check out the [ROADMAP](ROADMAP.md) for the full scope of the project!
