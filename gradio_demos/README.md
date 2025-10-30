# Gradio NLP Demos

Interactive web interfaces for NLP models using Gradio.

## Overview

This directory contains Gradio-based web applications for various NLP tasks. Gradio makes it easy to create beautiful, shareable demos for machine learning models.

## Features

- ✅ Interactive web interfaces
- ✅ No frontend coding required
- ✅ Easy to share and deploy
- ✅ Built-in examples
- ✅ File upload support
- ✅ Multiple input/output types
- ✅ Custom themes

## Available Demos

### 1. Sentiment Analysis
Interactive sentiment analysis with BERT.

### 2. Text Classification
Multi-class text classification demo.

### 3. Named Entity Recognition
Visualize entities in text.

### 4. Text Summarization
Generate summaries of long texts.

### 5. Question Answering
Ask questions about a context.

## Quick Start

### Installation

```bash
pip install gradio transformers torch
```

### Run a Demo

```bash
# Sentiment analysis
python -m gradio_demos.sentiment_analysis

# Text classification
python -m gradio_demos.text_classification

# NER
python -m gradio_demos.ner_demo
```

## Creating Custom Demos

### Basic Structure

```python
import gradio as gr

def predict(text):
    # Your model prediction logic
    return result

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Label(label="Prediction"),
    title="My NLP Demo",
    description="Enter text to classify",
)

# Launch
demo.launch()
```

### Advanced Features

```python
import gradio as gr

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Text", lines=5),
        gr.Slider(0, 1, value=0.5, label="Confidence Threshold"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.JSON(label="Detailed Results"),
    ],
    examples=[
        ["This is a great product!"],
        ["Terrible service, very disappointed."],
    ],
    title="Advanced Demo",
    theme="huggingface",
    allow_flagging="never",
)
```

## Deployment

### Local
```bash
python app.py
```

### Hugging Face Spaces
```bash
# Push to Hugging Face
git add .
git commit -m "Add Gradio demo"
git push origin main
```

### Docker
```bash
docker build -t gradio-demo .
docker run -p 7860:7860 gradio-demo
```

## Examples

### Sentiment Analysis
```python
import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    return {result["label"]: result["score"]}

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text here..."),
    outputs=gr.Label(num_top_classes=2),
    examples=[
        "I love this!",
        "This is terrible.",
        "It's okay, nothing special.",
    ],
)

demo.launch()
```

### Batch Processing
```python
def process_batch(file):
    # Read file
    texts = file.read().decode().split('\n')
    
    # Process each text
    results = [predict(text) for text in texts]
    
    return results

demo = gr.Interface(
    fn=process_batch,
    inputs=gr.File(label="Upload text file"),
    outputs=gr.JSON(label="Results"),
)
```

## Customization

### Themes
```python
demo = gr.Interface(
    ...,
    theme=gr.themes.Soft(),
    # Or: "default", "huggingface", "glass", "monochrome"
)
```

### Custom CSS
```python
demo = gr.Interface(
    ...,
    css="""
        .gradio-container {
            font-family: 'Arial';
        }
    """
)
```

## Tips

1. **Use examples** to guide users
2. **Add clear descriptions** for inputs/outputs
3. **Enable caching** for faster responses
4. **Add error handling** for robustness
5. **Use progress bars** for long operations

## Troubleshooting

**Port already in use:**
```python
demo.launch(server_port=7861)
```

**Share publicly:**
```python
demo.launch(share=True)
```

**Debug mode:**
```python
demo.launch(debug=True)
```

## Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Gradio Gallery](https://gradio.app/demos/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

## Contributing

Contributions are welcome! Add new demos or improve existing ones.

## License

Part of the EudaLabs NLP repository.
