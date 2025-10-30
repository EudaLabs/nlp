"""Gradio demo for sentiment analysis."""
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class SentimentAnalyzer:
    """Sentiment analysis model wrapper."""
    
    def __init__(self, model_path: str = None):
        """Initialize the sentiment analyzer."""
        if model_path and os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Use a pre-trained model from Hugging Face
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = ["Negative 😞", "Positive 😊"]
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment of text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with label probabilities
        """
        if not text or not text.strip():
            return {}
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        # Format results
        results = {
            self.labels[i]: float(probabilities[0][i])
            for i in range(len(self.labels))
        }
        
        return results


# Initialize model
analyzer = SentimentAnalyzer()


def analyze_text(text: str) -> dict:
    """Analyze sentiment of input text."""
    return analyzer.predict(text)


# Create Gradio interface
demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text to analyze sentiment...",
        lines=5,
    ),
    outputs=gr.Label(
        label="Sentiment Prediction",
        num_top_classes=2,
    ),
    title="🎭 Sentiment Analysis",
    description="""
    Analyze the sentiment of any text using a fine-tuned BERT model.
    
    **How it works:**
    1. Enter any text (review, tweet, comment, etc.)
    2. The model analyzes the sentiment
    3. Get confidence scores for positive/negative sentiment
    
    Try the examples below or enter your own text!
    """,
    examples=[
        ["This movie is absolutely fantastic! Best film I've seen this year."],
        ["Terrible product, complete waste of money. Very disappointed."],
        ["It was okay, nothing special but not bad either."],
        ["I love this! Highly recommended to everyone!"],
        ["The worst experience of my life. Never again."],
        ["Pretty good overall, would consider buying again."],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
