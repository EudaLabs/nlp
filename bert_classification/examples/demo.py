"""Demo script showcasing BERT classification usage."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bert_classification.inference import BERTClassifier


def demo_sentiment_analysis():
    """Demo sentiment analysis with pre-trained model."""
    print("=" * 70)
    print("BERT Sentiment Analysis Demo")
    print("=" * 70)
    
    # Note: This assumes you have a trained model
    # To train a model first, run:
    # python -m bert_classification.train --dataset imdb --epochs 1
    
    model_path = "./models/bert-imdb"
    
    try:
        classifier = BERTClassifier(model_path, num_classes=2)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease train a model first:")
        print("python -m bert_classification.train --dataset imdb --epochs 1")
        return
    
    # Test texts
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible movie, complete waste of time.",
        "It was okay, nothing special but not bad either.",
        "One of the best films I've ever seen! Highly recommended.",
        "I fell asleep halfway through, very boring.",
    ]
    
    print("\nPredictions:")
    print("-" * 70)
    
    labels = ["Negative", "Positive"]
    
    for text in test_texts:
        result = classifier.predict(text)
        label = labels[result["label"]]
        confidence = result["confidence"]
        
        print(f"\nText: {text}")
        print(f"Sentiment: {label} (Confidence: {confidence:.2%})")


def demo_custom_classification():
    """Demo custom classification example."""
    print("\n" + "=" * 70)
    print("Custom Classification Example")
    print("=" * 70)
    
    print("\nFor custom classification tasks:")
    print("1. Prepare your dataset (texts and labels)")
    print("2. Train the model:")
    
    example_code = """
    from bert_classification.train import train_classifier
    
    # Your data
    train_texts = ["text1", "text2", ...]
    train_labels = [0, 1, ...]  # Integer labels
    
    val_texts = ["val_text1", ...]
    val_labels = [0, ...]
    
    # Train
    model, tokenizer, stats = train_classifier(
        texts=train_texts,
        labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        num_classes=3,  # Your number of classes
        epochs=5,
        batch_size=16,
        output_dir="./models/my-classifier"
    )
    """
    
    print(example_code)
    
    print("\n3. Use for inference:")
    
    inference_code = """
    from bert_classification.inference import BERTClassifier
    
    classifier = BERTClassifier("./models/my-classifier", num_classes=3)
    result = classifier.predict("your text here")
    print(result)
    """
    
    print(inference_code)


def demo_batch_prediction():
    """Demo batch prediction."""
    print("\n" + "=" * 70)
    print("Batch Prediction Example")
    print("=" * 70)
    
    batch_code = """
    from bert_classification.inference import BERTClassifier
    
    classifier = BERTClassifier("./models/bert-imdb")
    
    # Process multiple texts efficiently
    texts = ["text1", "text2", "text3", ...]
    results = classifier.predict_batch(texts, batch_size=32)
    
    for text, result in zip(texts, results):
        print(f"{text}: {result['label']} ({result['confidence']:.2%})")
    """
    
    print("\nEfficient batch processing:")
    print(batch_code)


if __name__ == "__main__":
    # Run demos
    demo_sentiment_analysis()
    demo_custom_classification()
    demo_batch_prediction()
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
