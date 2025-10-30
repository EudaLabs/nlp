# Named Entity Recognition (NER) System

Advanced Named Entity Recognition using multiple approaches: SpaCy, BERT, and custom training.

## Overview

This project provides comprehensive tools for Named Entity Recognition (NER), including:
- Pre-trained model usage (SpaCy, Hugging Face)
- Custom NER model training
- Entity visualization
- Multi-language support
- Evaluation metrics

## Features

- ✅ Multiple NER backends (SpaCy, BERT, custom)
- ✅ Pre-trained models for quick usage
- ✅ Custom training pipeline
- ✅ Entity visualization with displaCy
- ✅ Batch processing
- ✅ Support for custom entity types
- ✅ Evaluation metrics (precision, recall, F1)
- ✅ Multi-language support

## Quick Start

### Installation

```bash
pip install spacy transformers torch
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from ner_system.recognizer import NERRecognizer

# Initialize recognizer
recognizer = NERRecognizer(backend="spacy")

# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = recognizer.extract_entities(text)

for entity in entities:
    print(f"{entity.text}: {entity.label_}")
# Output:
# Apple Inc.: ORG
# Steve Jobs: PERSON
# Cupertino: GPE
# California: GPE
```

## Supported Entity Types

### Standard NER Labels

- **PERSON** - People, including fictional
- **ORG** - Organizations, companies, agencies
- **GPE** - Geo-political entities (countries, cities, states)
- **LOC** - Non-GPE locations (mountains, bodies of water)
- **DATE** - Dates or periods
- **TIME** - Times smaller than a day
- **MONEY** - Monetary values
- **PERCENT** - Percentages
- **PRODUCT** - Products, vehicles, foods
- **EVENT** - Named events
- **WORK_OF_ART** - Titles of books, songs, etc.
- **LAW** - Named documents made into laws
- **LANGUAGE** - Any named language

## Backends

### 1. SpaCy Backend (Fast, Accurate)

```python
recognizer = NERRecognizer(backend="spacy", model="en_core_web_sm")
entities = recognizer.extract_entities(text)
```

Models available:
- `en_core_web_sm` - Small model (fast)
- `en_core_web_md` - Medium model (balanced)
- `en_core_web_lg` - Large model (accurate)
- `en_core_web_trf` - Transformer-based (most accurate)

### 2. BERT Backend (Transformer-based)

```python
recognizer = NERRecognizer(backend="bert")
entities = recognizer.extract_entities(text)
```

Uses: `dslim/bert-base-NER` by default

### 3. Custom Training

```python
from ner_system.trainer import NERTrainer

# Prepare training data
train_data = [
    ("Apple is looking at buying U.K. startup", 
     {"entities": [(0, 5, "ORG"), (27, 31, "GPE")]}),
    # More examples...
]

# Train custom model
trainer = NERTrainer()
model = trainer.train(train_data, epochs=10)
```

## Visualization

```python
from ner_system.visualizer import visualize_entities

# Visualize in Jupyter/browser
html = visualize_entities(text, entities)

# Save to file
visualize_entities(text, entities, output_file="entities.html")
```

## Batch Processing

```python
texts = [
    "Text 1 with entities...",
    "Text 2 with entities...",
    # ...
]

results = recognizer.batch_extract(texts)
```

## Evaluation

```python
from ner_system.evaluator import NEREvaluator

# Prepare test data
test_data = [
    ("Text", {"entities": [(start, end, label), ...]}),
    # More examples...
]

# Evaluate
evaluator = NEREvaluator(recognizer)
metrics = evaluator.evaluate(test_data)

print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")
```

## Advanced Usage

### Custom Entity Types

```python
# Define custom entity types
custom_types = ["TECH_PRODUCT", "COMPANY", "FOUNDER"]

# Train on custom data
trainer = NERTrainer(entity_types=custom_types)
model = trainer.train(custom_train_data)
```

### Multi-language Support

```python
# German NER
recognizer = NERRecognizer(backend="spacy", model="de_core_news_sm")

# French NER
recognizer = NERRecognizer(backend="spacy", model="fr_core_news_sm")

# Spanish NER
recognizer = NERRecognizer(backend="spacy", model="es_core_news_sm")
```

### Entity Linking

```python
# Extract and link entities to knowledge bases
from ner_system.linker import EntityLinker

linker = EntityLinker()
linked_entities = linker.link_entities(entities)

for entity in linked_entities:
    print(f"{entity.text}: {entity.wikipedia_url}")
```

## CLI Usage

```bash
# Extract entities from text
python -m ner_system.cli extract "Apple was founded in California"

# Process a file
python -m ner_system.cli extract-file input.txt --output entities.json

# Visualize entities
python -m ner_system.cli visualize "Text with entities" --output viz.html

# Train custom model
python -m ner_system.cli train --data train.json --output ./models/custom-ner
```

## API Integration

```python
from fastapi import FastAPI
from ner_system.api import create_ner_api

app = create_ner_api()

# POST /extract
# {
#   "text": "Apple was founded in California"
# }
```

## Performance Benchmarks

| Backend | Speed (texts/sec) | F1 Score | Model Size |
|---------|------------------|----------|------------|
| SpaCy (sm) | 1000+ | 0.85 | 12 MB |
| SpaCy (md) | 500+ | 0.88 | 40 MB |
| SpaCy (lg) | 200+ | 0.90 | 560 MB |
| SpaCy (trf) | 50+ | 0.92 | 440 MB |
| BERT | 100+ | 0.91 | 420 MB |

## Use Cases

### 1. Information Extraction
Extract structured information from unstructured text.

### 2. Document Classification
Use entities to categorize documents.

### 3. Knowledge Graph Construction
Build knowledge graphs from entity relationships.

### 4. Privacy Compliance
Identify and redact PII (Personally Identifiable Information).

### 5. Search Enhancement
Improve search with entity-based indexing.

## Examples

### Example 1: News Article Processing

```python
news = """
Tesla CEO Elon Musk announced on Twitter that the company will 
open a new factory in Berlin, Germany next year. The $5 billion 
investment will create 10,000 jobs.
"""

entities = recognizer.extract_entities(news)

# Group by type
from collections import defaultdict
grouped = defaultdict(list)
for ent in entities:
    grouped[ent.label_].append(ent.text)

print("Organizations:", grouped["ORG"])
print("People:", grouped["PERSON"])
print("Locations:", grouped["GPE"])
```

### Example 2: Resume Parsing

```python
resume = """
John Doe
Software Engineer at Google
Experience: Python, TensorFlow, AWS
Education: Stanford University, Computer Science
"""

# Extract skills and education
entities = recognizer.extract_entities(resume)
skills = [e.text for e in entities if e.label_ == "SKILL"]
education = [e.text for e in entities if e.label_ == "ORG"]
```

### Example 3: Medical Text Processing

```python
# Use BioBERT for medical NER
recognizer = NERRecognizer(
    backend="bert",
    model="dmis-lab/biobert-base-cased-v1.1"
)

medical_text = """
Patient presents with hypertension and type 2 diabetes.
Prescribed metformin 500mg twice daily.
"""

entities = recognizer.extract_entities(medical_text)
```

## Troubleshooting

**Model not found:**
```bash
python -m spacy download en_core_web_sm
```

**Low accuracy:**
- Use larger models (md, lg, trf)
- Train custom model on domain data
- Increase training data

**Slow performance:**
- Use smaller models (sm)
- Enable batch processing
- Use GPU acceleration

## Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- More entity types
- Improved visualization
- Performance optimizations

## References

- [SpaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- [Hugging Face NER](https://huggingface.co/tasks/token-classification)
- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)

## License

Part of the EudaLabs NLP repository.
