"""Inference script for BERT text classification."""
import argparse
from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import get_device, logger


class BERTClassifier:
    """BERT classifier for inference."""
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 2,
        device: str = None,
    ):
        """
        Initialize BERT classifier.
        
        Args:
            model_path: Path to the trained model
            num_classes: Number of classes
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.num_classes = num_classes
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict class for input text(s).
        
        Args:
            text: Single text or list of texts
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dictionary or list of dictionaries with predictions
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        
        # Get predictions
        if return_probabilities:
            probabilities = torch.softmax(logits, dim=-1)
            probs = probabilities.cpu().numpy()
        
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "label": int(pred),
                "confidence": float(probs[i][pred]) if return_probabilities else None,
            }
            
            if return_probabilities and self.num_classes <= 10:
                result["probabilities"] = {
                    f"class_{j}": float(probs[i][j])
                    for j in range(self.num_classes)
                }
            
            results.append(result)
        
        # Return single result or list
        return results[0] if single_input else results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Predict classes for a batch of texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.predict(batch_texts)
            all_results.extend(batch_results)
        
        return all_results


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(description="BERT text classification inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--texts-file", type=str, help="File with texts (one per line)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = BERTClassifier(
        model_path=args.model_path,
        num_classes=args.num_classes,
    )
    
    # Single text
    if args.text:
        result = classifier.predict(args.text)
        logger.info(f"\nInput: {args.text}")
        logger.info(f"Prediction: {result}")
    
    # Multiple texts from file
    elif args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(texts)} texts...")
        results = classifier.predict_batch(texts, batch_size=args.batch_size)
        
        for i, (text, result) in enumerate(zip(texts, results)):
            logger.info(f"\n[{i+1}] {text[:100]}...")
            logger.info(f"    Prediction: {result}")
    
    else:
        logger.error("Please provide either --text or --texts-file")


if __name__ == "__main__":
    main()
