"""Gradio demo for multi-class text classification."""
import gradio as gr
import torch
from transformers import pipeline


class TextClassifier:
    """Text classifier using zero-shot classification."""
    
    def __init__(self):
        """Initialize the classifier."""
        # Use zero-shot classification for demo purposes
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
        )
        
        self.default_labels = [
            "Technology",
            "Sports",
            "Politics",
            "Entertainment",
            "Business",
            "Science",
            "Health",
        ]
    
    def predict(self, text: str, labels: str) -> dict:
        """
        Classify text into one of the provided labels.
        
        Args:
            text: Input text to classify
            labels: Comma-separated list of labels
        
        Returns:
            Dictionary with label probabilities
        """
        if not text or not text.strip():
            return {}
        
        # Parse labels
        label_list = [l.strip() for l in labels.split(",") if l.strip()]
        if not label_list:
            label_list = self.default_labels
        
        # Classify
        result = self.classifier(text, label_list)
        
        # Format results
        return {
            label: score
            for label, score in zip(result["labels"], result["scores"])
        }


# Initialize classifier
classifier = TextClassifier()


def classify_text(text: str, labels: str = None) -> dict:
    """Classify text into categories."""
    if not labels:
        labels = ", ".join(classifier.default_labels)
    return classifier.predict(text, labels)


# Create Gradio interface
demo = gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter text to classify...",
            lines=5,
        ),
        gr.Textbox(
            label="Categories (comma-separated)",
            placeholder="Technology, Sports, Politics, ...",
            value=", ".join(classifier.default_labels),
        ),
    ],
    outputs=gr.Label(
        label="Classification Results",
        num_top_classes=5,
    ),
    title="📝 Multi-Class Text Classification",
    description="""
    Classify text into custom categories using zero-shot classification.
    
    **How it works:**
    1. Enter any text you want to classify
    2. Provide categories (or use the defaults)
    3. Get confidence scores for each category
    
    **Zero-shot classification** means the model can classify into categories it wasn't explicitly trained on!
    """,
    examples=[
        [
            "Apple announces new iPhone with revolutionary AI features",
            "Technology, Sports, Politics, Entertainment, Business",
        ],
        [
            "The Lakers won the championship game last night",
            "Technology, Sports, Politics, Entertainment, Business",
        ],
        [
            "Scientists discover new species in the Amazon rainforest",
            "Science, Technology, Health, Environment, News",
        ],
        [
            "The new Marvel movie breaks box office records",
            "Entertainment, Business, Sports, News, Culture",
        ],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
